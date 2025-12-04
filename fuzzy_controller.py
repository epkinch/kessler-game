import os
import math
import argparse

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import EasyGA as ga

# Only imported for type hinting
from EasyGA.structure import Chromosome
from kesslergame.state_models import GameState, ShipState, AsteroidView

from kesslergame import KesslerController, Scenario, TrainerEnvironment, GraphicsType


BULLET_TIME_UNIVERSE  = np.linspace(0, 2.0, 1001)
THETA_DELTA_UNIVERSE  = np.linspace(-math.pi, math.pi, 181)
THREAT_LEVEL_UNIVERSE = np.linspace(0, 1.0, 11)
SHIP_THRUST_UNIVERSE  = np.linspace(-480.0, 480.0, 100)
SHIP_TURN_UNIVERSE    = np.linspace(-180.0, 180.0, 361)


class FuzzyController(KesslerController):
    def __init__(self, chromosome: Chromosome = None):
        """
        """
        self.eval_frames = 0

        # Input variables
        # ===============
        # Bullet time is the amount of time the bullet needs to reach its
        # target and is also an approximation for the distance to the asteroid.
        bullet_time = ctrl.Antecedent(BULLET_TIME_UNIVERSE, 'bullet_time')
        # Theta delta is the amount the ship needs to turn to complete its
        # next action.
        theta_delta = ctrl.Antecedent(THETA_DELTA_UNIVERSE, 'theta_delta')
        # Threat level is a threat detection number that combines the distance,
        # size, and velocity of asteroids along with taking into account any
        # asteroid that will crash into the ship.
        threat_level = ctrl.Antecedent(THREAT_LEVEL_UNIVERSE, 'threat_level')

        # Output variables
        # ================
        ship_thrust = ctrl.Consequent(SHIP_THRUST_UNIVERSE,      'ship_thrust')
        ship_turn   = ctrl.Consequent(SHIP_TURN_UNIVERSE,        'ship_turn')
        ship_fire   = ctrl.Consequent(np.arange(-1.0, 1.0, 0.1), 'ship_fire')
        ship_mine   = ctrl.Consequent(np.arange(-1.0, 1.0, 0.1), 'ship_mine')

        if os.path.isfile(SOLUTION_PATH):
            with open(SOLUTION_PATH, 'r') as file:
                lines = file.readlines()
                training_data = lines[0].split(", ")
                print(f"Solution exists in solution.ga whose fitness is {training_data[0]} (fitness is between 0.0 and 1.0).")
                print(f"This was obtained using a population size of {training_data[1]} and a generation goal of {training_data[2]}.")
                chromosome = Chromosome([float(value) for value in lines[1:]])

        # Fallback values for if the genetic algorithm has not been run.
        if chromosome is None:
            # Bullet time sets
            bullet_time['VS'] = fuzz.trimf(bullet_time.universe, [0.0, 0.0, 0.2])
            bullet_time['S']  = fuzz.trimf(bullet_time.universe, [0.0, 0.2, 0.5])
            bullet_time['M']  = fuzz.trimf(bullet_time.universe, [0.2, 0.5, 1.0])
            bullet_time['L']  = fuzz.smf(bullet_time.universe,    0.5, 1.0)

            # Theta delta sets, we use zmf and smf to account for all angles
            # larger than 6° because this is the largest angle we can move in
            # a single game update.
            theta_delta['NL'] = fuzz.zmf(theta_delta.universe,    -math.pi/30, -math.pi/45)
            theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi/30, -math.pi/45, -math.pi/90])
            theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-math.pi/45, -math.pi/90,  math.pi/90])
            theta_delta['Z']  = fuzz.trimf(theta_delta.universe, [-math.pi/90,  0,           math.pi/90])
            theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-math.pi/90,  math.pi/90,  math.pi/45])
            theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [ math.pi/90,  math.pi/45,  math.pi/30])
            theta_delta['PL'] = fuzz.smf(theta_delta.universe,     math.pi/45,  math.pi/30)

            threat_level['L']  = fuzz.trimf(threat_level.universe, [0.0,  0.0, 0.25])
            threat_level['M']  = fuzz.trimf(threat_level.universe, [0.0,  0.3, 0.6])
            threat_level['H']  = fuzz.trimf(threat_level.universe, [0.4,  0.7, 1.0])
            threat_level['VH'] = fuzz.trimf(threat_level.universe, [0.75, 1.0, 1.0])

            # Output sets for thrust, due to the drag coefficient, any acceleration
            # below 80.0 m/s^2 will cause the ship to have zero acceleration, since
            # net acceleration is thrust + drag, where drag = 80 m/s^2. See ship.py
            ship_thrust['Reverse']     = fuzz.trimf(ship_thrust.universe, [-500.0, -500.0, -300.0])
            ship_thrust['SlowReverse'] = fuzz.trimf(ship_thrust.universe, [-400.0, -225.0,  -50.0])
            ship_thrust['Zero']        = fuzz.trimf(ship_thrust.universe, [ -80.0,    0.0,   80.0])
            ship_thrust['SlowForward'] = fuzz.trimf(ship_thrust.universe, [  50.0,  225.0,  400.0])
            ship_thrust['Forward']     = fuzz.trimf(ship_thrust.universe, [ 300.0,  500.0,  500.0])

            # Output sets for turn rate
            ship_turn['HardRight'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
            ship_turn['MedRight']  = fuzz.trimf(ship_turn.universe, [-180, -120,  -60])
            ship_turn['Right']     = fuzz.trimf(ship_turn.universe, [-120,  -60,   60])
            ship_turn['Zero']      = fuzz.trimf(ship_turn.universe, [ -60,    0,   60])
            ship_turn['Left']      = fuzz.trimf(ship_turn.universe, [ -60,   60,  120])
            ship_turn['MedLeft']   = fuzz.trimf(ship_turn.universe, [  60,  120,  180])
            ship_turn['HardLeft']  = fuzz.trimf(ship_turn.universe, [ 120,  180,  180])
        else:
            bullet_time['VS'] = fuzz.trimf(bullet_time.universe, chromosome[0:3])
            bullet_time['S']  = fuzz.trimf(bullet_time.universe, chromosome[3:6])
            bullet_time['M']  = fuzz.trimf(bullet_time.universe, chromosome[6:9])
            bullet_time['L']  = fuzz.smf(bullet_time.universe,   chromosome[9], chromosome[10])

            theta_delta['NL'] = fuzz.zmf(theta_delta.universe,   chromosome[11], chromosome[12])
            theta_delta['NM'] = fuzz.trimf(theta_delta.universe, chromosome[13:16])
            theta_delta['NS'] = fuzz.trimf(theta_delta.universe, chromosome[16:19])
            theta_delta['Z']  = fuzz.trimf(theta_delta.universe, chromosome[19:22])
            theta_delta['PS'] = fuzz.trimf(theta_delta.universe, chromosome[22:25])
            theta_delta['PM'] = fuzz.trimf(theta_delta.universe, chromosome[25:28])
            theta_delta['PL'] = fuzz.smf(theta_delta.universe,   chromosome[28], chromosome[29])

            threat_level['L']  = fuzz.trimf(threat_level.universe, chromosome[30:33])
            threat_level['M']  = fuzz.trimf(threat_level.universe, chromosome[33:36])
            threat_level['H']  = fuzz.trimf(threat_level.universe, chromosome[36:39])
            threat_level['VH'] = fuzz.trimf(threat_level.universe, chromosome[39:42])

            ship_thrust['Reverse']     = fuzz.trimf(ship_thrust.universe, chromosome[42:45])
            ship_thrust['SlowReverse'] = fuzz.trimf(ship_thrust.universe, chromosome[45:48])
            ship_thrust['Zero']        = fuzz.trimf(ship_thrust.universe, chromosome[48:51])
            ship_thrust['SlowForward'] = fuzz.trimf(ship_thrust.universe, chromosome[51:54])
            ship_thrust['Forward']     = fuzz.trimf(ship_thrust.universe, chromosome[54:57])

            ship_turn['HardRight'] = fuzz.trimf(ship_turn.universe, chromosome[57:60])
            ship_turn['MedRight']  = fuzz.trimf(ship_turn.universe, chromosome[60:63])
            ship_turn['Right']     = fuzz.trimf(ship_turn.universe, chromosome[63:66])
            ship_turn['Zero']      = fuzz.trimf(ship_turn.universe, chromosome[66:69])
            ship_turn['Left']      = fuzz.trimf(ship_turn.universe, chromosome[69:72])
            ship_turn['MedLeft']   = fuzz.trimf(ship_turn.universe, chromosome[72:75])
            ship_turn['HardLeft']  = fuzz.trimf(ship_turn.universe, chromosome[75:78])

        # Output sets for fire and mine
        ship_fire['Yes'] = fuzz.trimf(ship_fire.universe, [ 0,  1, 1])
        ship_fire['No']  = fuzz.trimf(ship_fire.universe, [-1, -1, 0])
        ship_mine['Yes'] = fuzz.trimf(ship_mine.universe, [ 0,  1, 1])
        ship_mine['No']  = fuzz.trimf(ship_mine.universe, [-1, -1, 0])

        # Fuzzy rules
        # ===========
        rules = []

        rules.append(ctrl.Rule((theta_delta['NM'] | theta_delta['NS']) & (threat_level['L'] | threat_level['M']), (ship_thrust['Zero'],        ship_turn['Right'],    ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule((theta_delta['PM'] | theta_delta['PS']) & (threat_level['L'] | threat_level['M']), (ship_thrust['Zero'],        ship_turn['Left'],     ship_fire['Yes'], ship_mine['No'])))

        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['NL'] & (threat_level['H'] | threat_level['VH']), (ship_thrust['SlowReverse'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['PL'] & (threat_level['H'] | threat_level['VH']), (ship_thrust['SlowReverse'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['NS'] & (threat_level['H'] | threat_level['VH']), (ship_thrust['SlowReverse'], ship_turn['HardRight'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['Z'] & (threat_level['H'] | threat_level['VH']),  (ship_thrust['SlowReverse'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['PS'] & (threat_level['H'] | threat_level['VH']), (ship_thrust['SlowReverse'], ship_turn['HardLeft'],  ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['NL'] & (threat_level['L'] | threat_level['M']),  (ship_thrust['SlowForward'], ship_turn['HardLeft'],  ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['VS'] & theta_delta['PL'] & (threat_level['L'] | threat_level['M']),  (ship_thrust['SlowForward'], ship_turn['HardRight'], ship_fire['No'],  ship_mine['No'])))

        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NM'],                                             (ship_thrust['SlowReverse'], ship_turn['MedRight'],  ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NS'],                                             (ship_thrust['SlowReverse'], ship_turn['Right'],     ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['Z'],                                              (ship_thrust['SlowReverse'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PS'],                                             (ship_thrust['SlowReverse'], ship_turn['Left'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PM'],                                             (ship_thrust['SlowReverse'], ship_turn['MedLeft'],   ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NL'] & (threat_level['L'] | threat_level['M']),   (ship_thrust['Zero'],        ship_turn['HardLeft'],  ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NL'] & (threat_level['H'] | threat_level['VH']),  (ship_thrust['Zero'],        ship_turn['HardRight'], ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PL'] & (threat_level['L'] | threat_level['M']),   (ship_thrust['Forward'],     ship_turn['HardLeft'],  ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PL'] & (threat_level['H'] | threat_level['VH']),  (ship_thrust['SlowForward'], ship_turn['HardLeft'],  ship_fire['No'],  ship_mine['No'])))

        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['NL'],                                             (ship_thrust['SlowForward'], ship_turn['HardRight'], ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['NM'],                                             (ship_thrust['SlowForward'], ship_turn['MedRight'],  ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['NS'],                                             (ship_thrust['SlowForward'], ship_turn['Right'],     ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['Z'],                                              (ship_thrust['SlowForward'], ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['PS'],                                             (ship_thrust['SlowForward'], ship_turn['Left'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['PM'],                                             (ship_thrust['SlowForward'], ship_turn['MedLeft'],   ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['PL'],                                             (ship_thrust['SlowForward'], ship_turn['HardLeft'],  ship_fire['No'],  ship_mine['No'])))

        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NL'],                                             (ship_thrust['SlowForward'], ship_turn['HardRight'], ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NM'],                                             (ship_thrust['Forward'],     ship_turn['MedRight'],  ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NS'],                                             (ship_thrust['Forward'],     ship_turn['Right'],     ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['Z'],                                              (ship_thrust['Forward'],     ship_turn['Zero'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PS'],                                             (ship_thrust['Forward'],     ship_turn['Left'],      ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PM'],                                             (ship_thrust['Forward'],     ship_turn['MedLeft'],   ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PL'],                                             (ship_thrust['SlowForward'], ship_turn['HardLeft'],  ship_fire['No'],  ship_mine['No'])))

        self.control_system = ctrl.ControlSystem(rules)


    def find_most_dangerous_asteroid(self, ship_state: ShipState, game_state: GameState) -> tuple[AsteroidView, float]:
        """Find the most dangerous asteroid based on distance, size, and velocity.

        Arguments:
            ship_state: The state of the ship to find the most dangerous
                asteroid for.
            game_state: The current state of the game containing information
                about all asteroids.

        Returns:
            (most_dangerous_asteroid, threat_level): A tuple containing the
                predicted most dangerous asteroid for the given ship based
                on the current state of the game and the threat level of
                this asteroid.
        """
        most_dangerous = None
        highest_threat = -1

        for asteroid in game_state["asteroids"]:
            ship_position = Vec2D(ship_state["position"])
            asteroid_position = Vec2D(asteroid["position"])
            asteroid_velocity = Vec2D(asteroid["velocity"])
            dist_vec = ship_position - asteroid_position
            angle_between = math.acos(min(dist_vec.dot_prod(asteroid_velocity) / (dist_vec.magnitude() * asteroid_velocity.magnitude()), 1.0))
            angle_between = angle_between * 180.0 / math.pi

            # Threat calculation: closer, larger, faster = more dangerous
            distance_threat = max(0, 1.0 - dist_vec.magnitude() / 1300.0)

            # Not guaranteed to hit the ship, but gives a good estimate on
            # which asteroids if a collision could happen
            if angle_between < 15.0:
                intercept_time = dist_vec.magnitude() / asteroid_velocity.magnitude()
                intercept_threat = max(0, 1 - intercept_time / 100.0)
                size_threat = asteroid["size"] / 4.0
                velocity_threat = min(1.0, asteroid_velocity.magnitude() / 200.0)

                # Combined threat score where we prioritize asteroids that will
                # intercept the ship.
                threat_score = (0.5 * intercept_threat + 0.4 * distance_threat + 0.075 * size_threat + 0.025 * velocity_threat)
            else:
                # When the asteroid won't hit the ship, it's best to only
                # prioritize the distance, otherwise the ship may prioritize
                # further away but larger asteroids that don't make sense
                # to target at that point in time.
                threat_score = distance_threat

            if threat_score > highest_threat:
                highest_threat = threat_score
                most_dangerous = asteroid

        return most_dangerous, highest_threat


    def calculate_intercept(self, ship_state: ShipState, asteroid: AsteroidView):
        """Finds the distance between the ship and given asteroid.

        Arguments:
            ship_state: The current state of the ship that will be used to
                determine the shooting angle.
            asteroid: The asteroid to be targeted by the given ship.

        Returns:
            (intercept_time, dist_between): A tuple containing the intercept
                time and distance between the ship and the given asteroid. If
                `use_ship` is `True`, then it is the amount of time it will to
                take for the asteroid to intersect with the ship, or 0.0 when
                it will not. If `use_ship` is `False`, then it is the amount of
                time it will take for a bullet to intersect the given asteroid.
        """
        ship_position     = Vec2D(ship_state["position"])
        ship_velocity     = Vec2D(ship_state["velocity"])
        asteroid_position = Vec2D(asteroid["position"])
        asteroid_velocity = Vec2D(asteroid["velocity"])

        # Determine the vector that points from the asteroid to the ship.
        asteroid_ship_vec = ship_position - asteroid_position
        asteroid_ship_theta = asteroid_ship_vec.direction()
        asteroid_direction = asteroid_velocity.direction()

        # Find the angle between the vector that points to the ship and
        # the asteroid's velocity vector. If we assume A = asteroid_ship_vec
        # and B = asteroid_velocity, then this is equivalent to:
        # intercept_angle = arccos(A * B / |A||B|)
        intercept_angle = asteroid_ship_theta - asteroid_direction

        # Apply the cosine to the value to cancel out the arccos operation
        cos_intercept = math.cos(intercept_angle)

        asteroid_vel = asteroid_velocity.magnitude()
        ship_ast_distance = asteroid_ship_vec.magnitude()
        obj_speed = 800.0

        # Applying the Law of Cosines on the triangle formed by the ship
        # velocity, asteroid velocity, and ship-asteroid distance vectors,
        # then transform into a quadratic equation at^2 + bt + c = 0.
        quadratic_coeff = asteroid_vel**2 - obj_speed**2
        linear_coeff = -2 * ship_ast_distance * asteroid_vel * cos_intercept
        const_term = ship_ast_distance**2
        targ_discriminant = linear_coeff**2 - 4*quadratic_coeff*const_term

        # No intersection occurs if the discriminant is negative.
        if targ_discriminant < 0 or quadratic_coeff == 0:
            return 0.0, ship_ast_distance

        sqrt_discriminant = math.sqrt(targ_discriminant)
        t1_intercept = (-linear_coeff + sqrt_discriminant) / (2 * quadratic_coeff)
        t2_intercept = (-linear_coeff - sqrt_discriminant) / (2 * quadratic_coeff)

        # Select the time intercept that is closer to zero, i.e., it happens
        # sooner in time than the other.
        if t1_intercept > t2_intercept:
            if t2_intercept >= 0:
                intercept_time = t2_intercept
            else:
                intercept_time = t1_intercept
        else:
            if t1_intercept >= 0:
                intercept_time = t1_intercept
            else:
                intercept_time = t2_intercept

        return intercept_time, ship_ast_distance


    def determine_turn_rate(self, intercept_time: float, ship_state: ShipState, asteroid: AsteroidView) -> float:
        ship_curr_position = Vec2D(ship_state["position"])
        ship_velocity      = Vec2D(ship_state["velocity"])
        ship_position      = ship_curr_position + ship_velocity * (1/30)
        asteroid_position  = Vec2D(asteroid["position"])
        asteroid_velocity  = Vec2D(asteroid["velocity"])
        intercept = asteroid_position + (intercept_time + 1/30) * asteroid_velocity
        ship_intercept_angle = math.atan2((intercept.y - ship_position.y), (intercept.x - ship_position.x))

        # The amount we need to turn the ship to aim at where we want to shoot
        shooting_theta = ship_intercept_angle - (math.pi/180 * ship_state["heading"])
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        return shooting_theta


    def actions(self, ship_state: ShipState, game_state: GameState) -> tuple[float, float, bool, bool]:
        """Main controller method called each time step when the game updates its states.

        Arguments:
            ship_state: The state of the ship being updated.
            game_state: The current state of the game.

        Returns:
            (`thrust`, `turn_rate`, `should_fire`, `should_drop_mine`): A tuple
                containing the new thrust rate and new turn rate of the ship
                along with two Boolean flags for if the ship should fire and if
                it should drop a mine.
        """
        # Find the most dangerous asteroid
        target_asteroid, threat_level = self.find_most_dangerous_asteroid(ship_state, game_state)

        if target_asteroid is None:
            return 100.0, 0.0, False, False

        # Calculate intercept parameters
        bullet_t, distance = self.calculate_intercept(ship_state, target_asteroid)
        shooting_theta = self.determine_turn_rate(bullet_t, ship_state, target_asteroid)

        # Create control system simulation
        controller = ctrl.ControlSystemSimulation(self.control_system, flush_after_run=1)

        controller.input['bullet_time'] = min(bullet_t, 0.99) if bullet_t else 1.0
        controller.input['theta_delta'] = shooting_theta
        controller.input['threat_level'] = threat_level

        try:
            controller.compute()
            thrust = float(controller.output['ship_thrust'])
            turn_rate = float(controller.output['ship_turn'])
            fire = bool(controller.output['ship_fire'] >= 0)
            drop_mine = bool(controller.output['ship_mine'] >= 0)

        except Exception as e:
            # Safe fallback behavior
            if distance < 200:  # Too close - evade
                thrust = float(-100.0)
                turn_rate = float(0.0)
                fire = bool(False)
                drop_mine = bool(True)
            elif bullet_t and bullet_t < 0.1:  # Good shot - take it
                thrust = float(0.0)
                turn_rate = float(0.0)
                fire = bool(True)
                drop_mine = bool(False)
            else:  # Default behavior
                thrust = float(100.0)
                turn_rate = float(0.0)
                fire = bool(False)
                drop_mine = bool(False)

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine
    
    @property
    def name(self) -> str:
        return "Fuzzy Controller"


class Vec2D:
    def __init__(self, components: tuple[int | float, int | float] = None, x: int | float = None, y: int | float = None):
        """Creates a vector from either a tuple or a set of x and y values.

        Either the tuple must be specified or the x and y values. If both are
        given, then a `ValueError` will be raised.

        Arguments:
            components: A tuple containing an x-value and a y-value representing
                the x and y components of the vector.
            x: The vector's x component value.
            y: The vector's y component value.

        Raises:
            ValueError: If both `a_tuple` and `x` or `y` are specified.
        """
        if components and (x or y):
            raise ValueError("Cannot provide both a tuple and coordinates to form a vector!")

        if components:
            self._x = components[0]
            self._y = components[1]
        else:
            self._x = x
            self._y = y


    @property
    def x(self) -> int | float:
        return self._x


    @property
    def y(self) -> int | float:
        return self._y


    def magnitude(self) -> float:
        """Returns the magnitude of the vector."""
        return math.sqrt(self._x**2 + self._y**2)


    def direction(self) -> float:
        """Returns the angle between the components of the vector.

        Measures the angle in the range [-pi, pi] relative to the positive
        x-axis. If the angle is below the x-axis then the angle is negative,
        otherwise it is positive.
        """
        return math.atan2(self._y, self.x)


    def dot_prod(self, other: Vec2D) -> int | float:
        return self._x * other._x + self._y * other._y


    def __mul__(self, other: int | float) -> Vec2D:
        """A new vector whose coordinates are scaled by the given `other` value.

        Arguments:
            other: A scalar value to scale the values of the vector.
        """
        return Vec2D(x=self._x * other, y=self._y * other)


    def __rmul__(self, other: int | float) -> Vec2D:
        """A new vector whose coordinates are scaled by the given `other` value.

        Arguments:
            other: A scalar value to scale the values of the vector.
        """
        return Vec2D(x=self._x * other, y=self._y * other)


    def __add__(self, other: Vec2D) -> Vec2D:
        """A new vector whose coordinates are sum of the vectors.

        Arguments:
            other: The vector to add to the vector's components.
        """
        return Vec2D(x=self._x + other._x, y=self._y + other._y)


    def __sub__(self, other: Vec2D) -> Vec2D:
        """A new vector whose coordinates are the difference of the vectors.

        Produces a vector whose coordinates are the difference between this
        vector's coordinates and the provided other vector's coordinates. If
        this vector is vector A and other is vector B, then the resulting
        vector will be the vector pointing from B to A.

        Arguments:
            other: The vector to find the difference between.
        """
        return Vec2D(x=self._x - other._x, y=self._y - other._y)


    def __str__(self) -> str:
        return f"<x={self._x:0.4f}, y={self._y:.4f}>"    


# =============================================================================
# Genetic Algorithm
# =============================================================================

def main(population_size: int, generations: int):
    """Runs a genetic algorithm that seeks to optimize the fuzzy sets for the Kessler game.

    Arguments:
        population_size: The size of the genetic algorithm's population in each
            generation.
        generations: The number of generations to evolve the algorithm over.
    """
    asteroids_ga = ga.GA()
    asteroids_ga.fitness_goal = 'max'
    asteroids_ga.population_size = population_size
    asteroids_ga.generation_goal = generations
    asteroids_ga.chromosome_length = 78
    asteroids_ga.fitness_function_impl = ga_fitness
    asteroids_ga.chromosome_impl = ga_chromosome

    best_chromosome = asteroids_ga.population[0]

    with open(SOLUTION_PATH, 'w') as file:
        file.write(str(best_chromosome.fitness) + "," + str(population_size) + ", " + str(generations) + '\n')
        for gene in best_chromosome:
            file.write(str(gene.value) + '\n')


def ga_fitness(chromosome: Chromosome) -> float:
    """Fitness function for the Kessler game.

    Fitness is determined by running a single game and measuring the
    ratio of the number of asteroids destroyed out of the total possible
    number of asteroids in the game (400), the accuracy ratio, and the ratio
    of the number of deaths out of the possible amount of deaths (3). These
    ratios are then combined in a weighted sum.
    """
    my_test_scenario = Scenario(name='Test Scenario',
                                num_asteroids=10,
                                ship_states=[
                                    {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                ],
                                map_size=(1000, 800),
                                time_limit=60,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)

    game = TrainerEnvironment()
    score, _ = game.run(scenario=my_test_scenario, controllers=[FuzzyController(chromosome)])
    team = score.teams[0]

    # We start with 10 asteroids with a size of 4, each can break into 3 smaller
    # asteroids until they reach size 1 giving us 10 * SUM(3^n) for n=0 to n = 3
    # asteroids which is 400 in total.
    fraction_asteroids = team.asteroids_hit / 400.0
    fraction_deaths = team.deaths / 3.0

    # These three terms are the most important to the game's score with the
    # number of asteroids being the most important and accuracy close by (for
    # breaking ties). Deaths are less important but still important for making
    # the game last long.
    return 0.45 * fraction_asteroids + 0.4 * team.accuracy + 0.15 * fraction_deaths


def ga_chromosome() -> list[float]:
    chromosome_data = []
    chromosome_data.extend(generate_bullet_mfs())
    chromosome_data.extend(generate_theta_delta_mfs())
    chromosome_data.extend(generate_thrust_mfs())
    chromosome_data.extend(generature_turn_mfs())
    return chromosome_data


def generate_bullet_mfs():
    min_point = float(min(BULLET_TIME_UNIVERSE))
    max_point = float(max(BULLET_TIME_UNIVERSE))
    vs_rightpoint = np.random.uniform(min_point, 0.4)
    vs_points = [min_point, min_point, vs_rightpoint]

    s_leftpoint = np.random.uniform(min_point, vs_rightpoint)
    s_rightpoint = np.random.uniform(vs_rightpoint, 0.8)
    s_midpoint = np.random.uniform(s_leftpoint, s_rightpoint)
    s_points = [s_leftpoint, s_midpoint, s_rightpoint]


def generate_theta_delta_mfs():
    pass

def generate_thrust_mfs():
    pass

def generature_turn_mfs():
    pass

# bullet_time['VS'] = fuzz.trimf(bullet_time.universe, [0.0, 0.0, 0.2])
# bullet_time['S']  = fuzz.trimf(bullet_time.universe, [0.0, 0.2, 0.5])
# bullet_time['M']  = fuzz.trimf(bullet_time.universe, [0.2, 0.5, 1.0])
# bullet_time['L']  = fuzz.smf(bullet_time.universe,    0.5, 1.0)

# theta_delta['NL'] = fuzz.zmf(theta_delta.universe,    -math.pi/30, -math.pi/45)
# theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi/30, -math.pi/45, -math.pi/90])
# theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-math.pi/45, -math.pi/90,  math.pi/90])
# theta_delta['Z']  = fuzz.trimf(theta_delta.universe, [-math.pi/90,  0,           math.pi/90])
# theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-math.pi/90,  math.pi/90,  math.pi/45])
# theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [ math.pi/90,  math.pi/45,  math.pi/30])
# theta_delta['PL'] = fuzz.smf(theta_delta.universe,     math.pi/45,  math.pi/30)

# threat_level['L']  = fuzz.trimf(threat_level.universe, [0.0,  0.0, 0.25])
# threat_level['M']  = fuzz.trimf(threat_level.universe, [0.0,  0.3, 0.6])
# threat_level['H']  = fuzz.trimf(threat_level.universe, [0.4,  0.7, 1.0])
# threat_level['VH'] = fuzz.trimf(threat_level.universe, [0.75, 1.0, 1.0])

# ship_thrust['Reverse']     = fuzz.trimf(ship_thrust.universe, [-500.0, -500.0, -300.0])
# ship_thrust['SlowReverse'] = fuzz.trimf(ship_thrust.universe, [-400.0, -225.0,  -50.0])
# ship_thrust['Zero']        = fuzz.trimf(ship_thrust.universe, [ -80.0,    0.0,   80.0])
# ship_thrust['SlowForward'] = fuzz.trimf(ship_thrust.universe, [  50.0,  225.0,  400.0])
# ship_thrust['Forward']     = fuzz.trimf(ship_thrust.universe, [ 300.0,  500.0,  500.0])

# ship_turn['HardRight'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
# ship_turn['MedRight']  = fuzz.trimf(ship_turn.universe, [-180, -120,  -60])
# ship_turn['Right']     = fuzz.trimf(ship_turn.universe, [-120,  -60,   60])
# ship_turn['Zero']      = fuzz.trimf(ship_turn.universe, [ -60,    0,   60])
# ship_turn['Left']      = fuzz.trimf(ship_turn.universe, [ -60,   60,  120])
# ship_turn['MedLeft']   = fuzz.trimf(ship_turn.universe, [  60,  120,  180])
# ship_turn['HardLeft']  = fuzz.trimf(ship_turn.universe, [ 120,  180,  180])



CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOLUTION_PATH = os.path.join(CURRENT_DIRECTORY, 'solution.ga')


if __name__ == "__main__":
    desc = ("A program that runs a genetic algorithm to find the best parameters for the Kessler Game."
            " Once the algorithm has been executed, a scenario can be executed using the fuzzy controller"
            " and it will obtain the parameters from the solution file.")
    parser = argparse.ArgumentParser(prog="group15_controller.py", description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--population", help="the number of individuals in the GA's population between 1 and 50", metavar="[1-50]", default=20, type=int)
    parser.add_argument("-g", "--generations", help="the number of generations to evolve the GA over between 1 and 10,000", metavar="[1-10000]", default=10, type=int)
    ns = parser.parse_args()

    if (ns.population < 1 or ns.population > 50) or (ns.generations < 1 or ns.generations > 10000):
        parser.print_usage()
    else:
        main(ns.population, ns.generations)