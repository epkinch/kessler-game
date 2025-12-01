from src.kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

class FuzzyController(KesslerController):
    def __init__(self):
        self.eval_frames = 0

        # Bullet time is the amount of time the bullet needs to reach its
        # target.
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')

        # Theta delta is the amount the ship needs to turn to complete its
        # next action which can vary from -pi to +pi.
        theta_delta = ctrl.Antecedent(np.linspace(-math.pi, math.pi, 20), 'theta_delta')

        # The following are for threat assessment
        # Asteroid max distance is dependent on the map size which is by
        # default 1000x800 giving a max distance on the diagonal.
        asteroid_distance = ctrl.Antecedent(np.arange(0, 1300, 10), 'asteroid_distance')

        # Asteroids can only be a size from 1 to 4
        asteroid_size = ctrl.Antecedent(np.arange(1, 5, 1), 'asteroid_size')

        # Initial max velocity of an asteroid is based on its size which is
        # 165.0 m/s for a size 1 asteroid, see asteroid.py.
        asteroid_velocity = ctrl.Antecedent(np.arange(0, 200, 10), 'asteroid_velocity')

        # Output variables
        ship_turn = ctrl.Consequent(np.arange(-180.0, 180.0, 1.0), 'ship_turn')
        ship_thrust = ctrl.Consequent(np.arange(-480.0, 480.0, 10.0), 'ship_thrust')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')
        ship_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_mine')

        # Bullet time sets
        bullet_time['VS'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.02])
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0.02, 0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0.02, 0.05, 0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.05, 0.1)

        # Theta delta sets
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -math.pi, -math.pi/2)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi, -math.pi/2, -math.pi/6])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-math.pi/3, -math.pi/6, 0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-math.pi/6, 0, math.pi/6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0, math.pi/6, math.pi/3])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/6, math.pi/2, math.pi])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, math.pi/2, math.pi)

        # Asteroid distance sets
        asteroid_distance['VClose'] = fuzz.zmf(asteroid_distance.universe, 0, 300)
        asteroid_distance['Close'] = fuzz.trimf(asteroid_distance.universe, [100, 400, 700])
        asteroid_distance['Medium'] = fuzz.trimf(asteroid_distance.universe, [350, 650, 950])
        asteroid_distance['Far'] = fuzz.trimf(asteroid_distance.universe, [600, 900, 1200])
        asteroid_distance['VFar'] = fuzz.smf(asteroid_distance.universe, 1000, 1300)

        # Asteroid size sets
        asteroid_size['Small'] = fuzz.trimf(asteroid_size.universe, [1, 1, 2])
        asteroid_size['Medium'] = fuzz.trimf(asteroid_size.universe, [1.5, 2.5, 3.5])
        asteroid_size['Large'] = fuzz.trimf(asteroid_size.universe, [3, 4, 4])

        # Asteroid velocity sets
        asteroid_velocity['Slow'] = fuzz.trimf(asteroid_velocity.universe, [0, 0, 100])
        asteroid_velocity['Medium'] = fuzz.trimf(asteroid_velocity.universe, [50, 100, 150])
        asteroid_velocity['Fast'] = fuzz.trimf(asteroid_velocity.universe, [100, 200, 200])

        # Output sets for turn rate
        ship_turn['HardLeft'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['Left'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        ship_turn['Zero'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['Right'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['HardRight'] = fuzz.trimf(ship_turn.universe, [120,180,180])

        # Output sets for thrust
        ship_thrust['Reverse'] = fuzz.trimf(ship_thrust.universe, [-500, -500, -400])
        ship_thrust['SlowReverse'] = fuzz.trimf(ship_thrust.universe, [-400, -300, -200])
        ship_thrust['Zero'] = fuzz.trimf(ship_thrust.universe, [-50, 0, 50])
        ship_thrust['SlowForward'] = fuzz.trimf(ship_thrust.universe, [200, 300, 400])
        ship_thrust['Forward'] = fuzz.trimf(ship_thrust.universe, [400, 500, 500])

        # Output sets for fire and mine
        ship_fire['No'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0])
        ship_fire['Yes'] = fuzz.trimf(ship_fire.universe, [0, 1, 1])

        ship_mine['No'] = fuzz.trimf(ship_mine.universe, [-1, -1, 0])
        ship_mine['Yes'] = fuzz.trimf(ship_mine.universe, [0, 1, 1])

        # Fuzzy rules
        rules = []

        # Very close, large, fast asteroids
        # rules.append(ctrl.Rule(asteroid_distance["VClose"] & ~asteroid_size["Small"] & ~asteroid_velocity["Slow"] & (theta_delta["NL"] | theta_delta["NM"]), (ship_turn["Right"], ship_thrust["Forward"], ship_fire["No"], ship_mine["Yes"])))
        # rules.append(ctrl.Rule(asteroid_distance["VClose"] & (theta_delta["PL"] | theta_delta["PM"]), (ship_turn["Left"], ship_thrust["Forward"], ship_fire["No"], ship_mine["Yes"])))
        # rules.append(ctrl.Rule(asteroid_distance["VClose"] & (theta_delta["NS"] | theta_delta["Z"] | theta_delta["PS"]), (ship_turn["HardRight"], ship_thrust["Forward"], ship_fire["No"], ship_mine["Yes"])))
        # rules.append(ctrl.Rule(asteroid_distance["VClose"] & (theta_delta["NS"] | theta_delta["Z"] | theta_delta["PS"]), (ship_turn["HardLeft"], ship_thrust["Forward"], ship_fire["No"], ship_mine["Yes"])))

        rules.append(ctrl.Rule(asteroid_distance['VClose'] & asteroid_size['Large']  & asteroid_velocity['Fast'],   (ship_turn['HardRight'], ship_thrust['Reverse'], ship_fire['Yes'], ship_mine['Yes'])))
        rules.append(ctrl.Rule(asteroid_distance['VClose'] & asteroid_size['Medium'] & asteroid_velocity['Fast'],   (ship_turn['Right'], ship_thrust['Reverse'], ship_fire['No'], ship_mine['Yes'])))
        rules.append(ctrl.Rule(asteroid_distance['VClose'] & asteroid_size['Large']  & asteroid_velocity['Medium'], (ship_turn['Left'], ship_thrust['Reverse'], ship_fire['No'], ship_mine['Yes'])))

        # Good shooting opportunities
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['Z'],  (ship_turn['Zero'],  ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['Left'],  ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['Right'], ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['Z'],  (ship_turn['Zero'],  ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['Left'],  ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['Right'], ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))

        # Rule Turning toward targets
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['HardLeft'],  ship_thrust['SlowForward'], ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['Left'],      ship_thrust['SlowForward'], ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['Left'],      ship_thrust['SlowForward'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['Right'],     ship_thrust['SlowForward'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['Right'],     ship_thrust['SlowForward'], ship_fire['No'],  ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['HardRight'], ship_thrust['SlowForward'], ship_fire['No'],  ship_mine['No'])))

        self.control_system = ctrl.ControlSystem(rules)


    def find_most_dangerous_asteroid(self, ship_state: Dict, game_state: Dict):
        """Find the most dangerous asteroid based on distance, size, and velocity.

        Arguments:
            ship_state: The state of the ship to find the most dangerous
                asteroid for.
            game_state: The current state of the game containing information
                about all asteroids.

        Returns:
            most_dangerous_asteroid: The predicted most dangerous asteroid for
                the given ship based on the current state of the game.
        """
        ship_position = Vec2D(ship_state["position"])
        most_dangerous = None
        highest_threat = -1

        for asteroid in game_state["asteroids"]:
            asteroid_position = Vec2D(asteroid["position"])
            asteroid_velocity = Vec2D(asteroid["velocity"])
            ship_asteroid_vec = ship_position - asteroid_position

            # Calculate distance
            distance = ship_asteroid_vec.magnitude()

            # Threat calculation: closer, larger, faster = more dangerous
            distance_threat = max(0, 1 - distance / 1300.0)
            size_threat = asteroid["size"] / 40
            velocity_threat = min(1.0, asteroid_velocity.magnitude() / 200.0)

            # Combined threat score where we prioritize the distance.
            threat_score = (0.6 * distance_threat + 0.3 * size_threat + 0.1 * velocity_threat)

            if threat_score > highest_threat:
                highest_threat = threat_score
                most_dangerous = asteroid

        return most_dangerous


    def calculate_intercept(self, ship_state: Dict, asteroid: Dict):
        """Finds the distance between the ship and given asteroid and the angle required to target it.

        Arguments:
            ship_state: The current state of the ship that will be used to
                determine the shooting angle.
            asteroid: The asteroid to be targeted by the given ship.

        Returns:
            (`bullet_time`, `shooting_theta`, `distance_between`, `asteroid_velocity`): A
                tuple containing the amount of time it will take for a bullet to reach the
                given asteroid, the angle the ship needs to rotate to target the asteroid,
                the distance between the ship and the asteroid, and the velocity of the
                asteroid. If the ship cannot target the given asteroid, then `bullet_time`
                and `shooting_theta` will be `None`.
        """
        ship_position     = Vec2D(ship_state["position"])
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

        # Bullets have a hardcoded speed of 800.0m/s. See bullet.py
        bullet_speed = 800.0

        # Applying the Law of Cosines on the triangle formed by the ship
        # velocity, asteroid velocity, and ship-asteroid distance vectors,
        # then transform into a quadratic equation at^2 + bt + c = 0.
        quadratic_coeff = asteroid_vel**2 - bullet_speed**2
        linear_coeff = -2 * ship_ast_distance * asteroid_vel * cos_intercept
        const_term = ship_ast_distance**2
        targ_discriminant = linear_coeff**2 - 4*quadratic_coeff*const_term

        # No intersection occurs if the discriminant is negative.
        if targ_discriminant < 0:
            return None, None, ship_ast_distance, asteroid_vel

        sqrt_discriminant = math.sqrt(targ_discriminant)
        t1_intercept = (-linear_coeff + sqrt_discriminant) / (2 * quadratic_coeff)
        t2_intercept = (-linear_coeff - sqrt_discriminant) / (2 * quadratic_coeff)

        # Select the time intercept that is closer to zero, i.e., it happens
        # sooner in time than the other.
        if t1_intercept > t2_intercept:
            if t2_intercept >= 0:
                bullet_t = t2_intercept
            else:
                bullet_t = t1_intercept
        else:
            if t1_intercept >= 0:
                bullet_t = t1_intercept
            else:
                bullet_t = t2_intercept

        intercept = asteroid_position + (bullet_t + 1/60) * asteroid_velocity
        ship_intercept_angle = math.atan2((intercept.y - ship_position.y), (intercept.x - ship_position.x))

        # The amount we need to turn the ship to aim at where we want to shoot
        shooting_theta = ship_intercept_angle - (math.pi/180 * ship_state["heading"])
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        return bullet_t, shooting_theta, ship_ast_distance, asteroid_vel


    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
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
        target_asteroid = self.find_most_dangerous_asteroid(ship_state, game_state)

        if target_asteroid is None:
            return 50.0, 0.0, False, False

        # Calculate intercept parameters
        bullet_t, shooting_theta, distance, asteroid_vel = self.calculate_intercept(ship_state, target_asteroid)

        # Create control system simulation
        controller = ctrl.ControlSystemSimulation(self.control_system, flush_after_run=1)

        controller.input['bullet_time'] = min(bullet_t, 0.99) if bullet_t else 1.0
        controller.input['theta_delta'] = shooting_theta
        controller.input['asteroid_distance'] = min(distance, 999)
        controller.input['asteroid_size'] = target_asteroid["size"]
        controller.input['asteroid_velocity'] = min(asteroid_vel, 399)

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
                turn_rate = float(90.0)
                fire = bool(False)
                drop_mine = bool(True)
            elif bullet_t and bullet_t < 0.1:  # Good shot - take it
                thrust = float(0.0)
                turn_rate = float(0.0)
                fire = bool(True)
                drop_mine = bool(False)
            else:  # Default behavior
                thrust = float(50.0)
                turn_rate = float(45.0)
                fire = bool(False)
                drop_mine = bool(False)

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine
    
    @property
    def name(self) -> str:
        return "Fuzzy Controller"

class Vec2D:
    def __init__(self, a_tuple: Tuple[int | float, int | float]=None, x=None, y=None):
        """Creates a vector from either a tuple or a set of x and y values.

        Either the tuple must be specified or the x and y values. If both are
        given, then a `ValueError` will be raised.

        Arguments:
            a_tuple: A tuple containing an x-value and a y-value representing
                the x and y components of the vector.
            x: The vector's x component value.
            y: The vector's y component value.

        Raises:
            ValueError: If both `a_tuple` and `x` or `y` are specified.
        """
        if a_tuple and (x or y):
            raise ValueError("Cannot provide both a tuple and coordinates to form a vector!")

        if a_tuple:
            self._x = a_tuple[0]
            self._y = a_tuple[1]
        else:
            self._x = x
            self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def magnitude(self):
        """Returns the magnitude of the vector based on its x and y components."""
        return math.sqrt(self._x**2 + self._y**2)

    def direction(self):
        """Returns the angle between the y and x components of the vector.

        Measures the angle in the range [-pi, pi] relative to the positive
        x-axis. If the angle is below the x-axis then the angle is negative,
        otherwise it is positive.
        """
        return math.atan2(self._y, self.x)

    def __mul__(self, other: Vec2D):
        return Vec2D(x=self._x * other, y=self._y * other)

    def __rmul__(self, other):
        return Vec2D(x=self._x * other, y=self._y * other)

    def __add__(self, other: Vec2D):
        return Vec2D(x=self._x + other._x, y=self._y + other._y)

    def __sub__(self, other: Vec2D):
        """A new vector whose coordinates are the difference of the vectors.

        Produces a vector whose coordinates are the difference between this
        vector's coordinates and the provided other vector's coordinates. If
        this vector is vector A and other is vector B, then the resulting
        vector will be the vector pointing from B to A.
        """
        return Vec2D(x=self._x - other._x, y=self._y - other._y)

    def __str__(self):
        return f"<x={self._x:0.4f}, y={self._y:.4f}>"