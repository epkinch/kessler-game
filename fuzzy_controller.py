from src.kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

class FuzzyController(KesslerController):
    def __init__(self):
        self.eval_frames = 0

        # Input variables
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi/30, math.pi/30, 0.1), 'theta_delta')

        # The following are for threat assessment
        asteroid_distance = ctrl.Antecedent(np.arange(0, 1000, 10), 'asteroid_distance')
        asteroid_size = ctrl.Antecedent(np.arange(10, 100, 5), 'asteroid_size')
        asteroid_velocity = ctrl.Antecedent(np.arange(0, 400, 10), 'asteroid_velocity')

        # Output variables
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_thrust = ctrl.Consequent(np.arange(-500, 500, 100), 'ship_thrust')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')
        ship_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_mine')

        # Bullet time sets
        bullet_time['VS'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.02]) # Very short
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0.02, 0.05]) # Short
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0.02, 0.05, 0.1]) # Medium
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.05, 0.1) # Long

        # Theta delta sets
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -math.pi, -math.pi/3)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi, -math.pi/3, -math.pi/6])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-math.pi/3, -math.pi/6, 0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-math.pi/6, 0, math.pi/6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0, math.pi/6, math.pi/3])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/6, math.pi/3, math.pi])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, math.pi/3, math.pi)

        # Asteroid distance sets
        asteroid_distance['VClose'] = fuzz.trimf(asteroid_distance.universe, [0, 0, 150])
        asteroid_distance['Close'] = fuzz.trimf(asteroid_distance.universe, [100, 200, 350])
        asteroid_distance['Medium'] = fuzz.trimf(asteroid_distance.universe, [250, 400, 600])
        asteroid_distance['Far'] = fuzz.trimf(asteroid_distance.universe, [450, 650, 900])
        asteroid_distance['VFar'] = fuzz.smf(asteroid_distance.universe, 700, 900)

        # Asteroid size sets
        asteroid_size['Small'] = fuzz.trimf(asteroid_size.universe, [10, 20, 40])
        asteroid_size['Medium'] = fuzz.trimf(asteroid_size.universe, [30, 50, 70])
        asteroid_size['Large'] = fuzz.smf(asteroid_size.universe, 60, 80)

        # Asteroid velocity sets
        asteroid_velocity['Slow'] = fuzz.trimf(asteroid_velocity.universe, [0, 0, 150])
        asteroid_velocity['Medium'] = fuzz.trimf(asteroid_velocity.universe, [100, 200, 300])
        asteroid_velocity['Fast'] = fuzz.smf(asteroid_velocity.universe, 250, 350)

        # Output sets for turn rate
        ship_turn['HardLeft'] = fuzz.trimf(ship_turn.universe, [-180, -180, -90])
        ship_turn['Left'] = fuzz.trimf(ship_turn.universe, [-180, -90, 0])
        ship_turn['Zero'] = fuzz.trimf(ship_turn.universe, [-45, 0, 45])
        ship_turn['Right'] = fuzz.trimf(ship_turn.universe, [0, 90, 180])
        ship_turn['HardRight'] = fuzz.trimf(ship_turn.universe, [90, 180, 180])

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
        rules.append(ctrl.Rule(asteroid_distance['VClose'] & asteroid_size['Large'] & asteroid_velocity['Fast'], 
                              (ship_turn['HardRight'], ship_thrust['Reverse'], ship_fire['No'], ship_mine['Yes'])))

        rules.append(ctrl.Rule(asteroid_distance['VClose'] & asteroid_size['Medium'] & asteroid_velocity['Fast'], 
                              (ship_turn['Right'], ship_thrust['Reverse'], ship_fire['No'], ship_mine['Yes'])))

        rules.append(ctrl.Rule(asteroid_distance['VClose'] & asteroid_size['Large'] & asteroid_velocity['Medium'], 
                              (ship_turn['Left'], ship_thrust['Reverse'], ship_fire['No'], ship_mine['Yes'])))

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
        ship_pos = ship_state["position"]
        ship_position = Vec2D(ship_pos[0], ship_pos[1])
        most_dangerous = None
        highest_threat = -1

        for asteroid in game_state["asteroids"]:
            asteroid_pos = asteroid["position"]
            asteroid_vel = asteroid["velocity"]
            asteroid_position = Vec2D(asteroid_pos[0], asteroid_pos[1])
            asteroid_velocity = Vec2D(asteroid_vel[0], asteroid_vel[1])
            ship_asteroid_vec = ship_position - asteroid_position

            # Calculate distance
            distance = ship_asteroid_vec.magnitude()

            # Threat calculation: closer, larger, faster = more dangerous
            distance_threat = max(0, 1 - distance / 800.0)
            size_threat = asteroid["size"] / 80
            velocity_threat = min(1.0, asteroid_velocity.magnitude() / 350.0)

            # Combined threat score
            threat_score = (0.5 * distance_threat + 0.3 * size_threat + 0.2 * velocity_threat)

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
        ship_pos_dict = ship_state["position"]
        asteroid_position_dict = asteroid["position"]
        asteroid_velocity_dict = asteroid["velocity"]
        ship_position     = Vec2D(ship_pos_dict[0], ship_pos_dict[1])
        asteroid_position = Vec2D(asteroid_position_dict[0], asteroid_position_dict[1])
        asteroid_velocity = Vec2D(asteroid_velocity_dict[0], asteroid_velocity_dict[1])

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
        """Main controller method called each time step"""

        # Find the most dangerous asteroid
        target_asteroid = self.find_most_dangerous_asteroid(ship_state, game_state)

        if target_asteroid is None:
            # No asteroids
            return 50.0, 45.0, False, False

        # Calculate intercept parameters
        bullet_t, shooting_theta, distance, asteroid_vel = self.calculate_intercept(ship_state, target_asteroid)

        # Create control system simulation
        controller = ctrl.ControlSystemSimulation(self.control_system, flush_after_run=1)

        # Set inputs
        controller.input['bullet_time'] = min(bullet_t, 0.99) if bullet_t else 1.0
        controller.input['theta_delta'] = shooting_theta
        controller.input['asteroid_distance'] = min(distance, 999)
        controller.input['asteroid_size'] = target_asteroid["size"]
        controller.input['asteroid_velocity'] = min(asteroid_vel, 399)

        try:
            # Compute outputs
            controller.compute()

            # Get defuzzified outputs
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

        # print(f"Thrust: {thrust}, Turn Rate: {turn_rate}, Fire: {fire}, Drop Mine: {drop_mine}")
        return thrust, turn_rate, fire, drop_mine
    
    @property
    def name(self) -> str:
        return "Fuzzy Controller"

class Vec2D:
    def __init__(self, x, y):
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
        return Vec2D(self._x * other, self._y * other)

    def __rmul__(self, other):
        return Vec2D(self._x * other, self._y * other)

    def __add__(self, other: Vec2D):
        return Vec2D(self._x + other._x, self._y + other._y)

    def __sub__(self, other: Vec2D):
        """A new vector whose coordinates are the difference of the vectors.

        Produces a vector whose coordinates are the difference between this
        vector's coordinates and the provided other vector's coordinates. If
        this vector is vector A and other is vector B, then the resulting
        vector will be the vector pointing from B to A.
        """
        return Vec2D(self._x - other._x, self._y - other._y)

    def __str__(self):
        return f"<x={self._x:0.4f}, y={self._y:.4f}>"