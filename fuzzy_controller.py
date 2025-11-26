from kesslergame import KesslerController
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
        ship_thrust = ctrl.Consequent(np.arange(-200, 200, 10), 'ship_thrust')
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
        ship_thrust['Reverse'] = fuzz.trimf(ship_thrust.universe, [-200, -200, -100])
        ship_thrust['SlowReverse'] = fuzz.trimf(ship_thrust.universe, [-150, -75, 0])
        ship_thrust['Zero'] = fuzz.trimf(ship_thrust.universe, [-50, 0, 50])
        ship_thrust['SlowForward'] = fuzz.trimf(ship_thrust.universe, [0, 75, 150])
        ship_thrust['Forward'] = fuzz.trimf(ship_thrust.universe, [100, 200, 200])
        
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
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['Z'], 
                              (ship_turn['Zero'], ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['NS'], 
                              (ship_turn['Left'], ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['S'] & theta_delta['PS'], 
                              (ship_turn['Right'], ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['Z'], 
                              (ship_turn['Zero'], ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['NS'], 
                              (ship_turn['Left'], ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['M'] & theta_delta['PS'], 
                              (ship_turn['Right'], ship_thrust['Zero'], ship_fire['Yes'], ship_mine['No'])))
        
        # Rule Turning toward targets
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NL'],
                              (ship_turn['HardLeft'], ship_thrust['SlowForward'], ship_fire['No'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NM'],
                              (ship_turn['Left'], ship_thrust['SlowForward'], ship_fire['No'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['NS'],
                              (ship_turn['Left'], ship_thrust['SlowForward'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PS'],
                              (ship_turn['Right'], ship_thrust['SlowForward'], ship_fire['Yes'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PM'],
                              (ship_turn['Right'], ship_thrust['SlowForward'], ship_fire['No'], ship_mine['No'])))
        rules.append(ctrl.Rule(bullet_time['L'] & theta_delta['PL'],
                              (ship_turn['HardRight'], ship_thrust['SlowForward'], ship_fire['No'], ship_mine['No'])))
        
        self.control_system = ctrl.ControlSystem(rules)
    
    def find_most_dangerous_asteroid(self, ship_state, game_state):
        """Find the most dangerous asteroid based on distance, size, and velocity"""
        ship_pos = ship_state["position"]
        most_dangerous = None
        highest_threat = -1
        
        for asteroid in game_state["asteroids"]:
            asteroid_pos = asteroid["position"]
            asteroid_vel = asteroid["velocity"]
            
            # Calculate distance
            distance = math.sqrt(
                (ship_pos[0] - asteroid_pos[0])**2 + 
                (ship_pos[1] - asteroid_pos[1])**2
            )
            
            # Calculate velocity magnitude
            velocity = math.sqrt(asteroid_vel[0]**2 + asteroid_vel[1]**2)
            
            # Threat calculation: closer, larger, faster = more dangerous
            distance_threat = max(0, 1 - distance / 800)
            size_threat = asteroid["size"] / 80
            velocity_threat = min(1.0, velocity / 350)
            
            # Combined threat score
            threat_score = (0.5 * distance_threat + 0.3 * size_threat + 0.2 * velocity_threat)
            
            if threat_score > highest_threat:
                highest_threat = threat_score
                most_dangerous = asteroid
        
        return most_dangerous
    
    # Based on scott_dick_controller.py
    def calculate_intercept(self, ship_state, asteroid):
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]
        asteroid_pos_x = asteroid["position"][0]
        asteroid_pos_y = asteroid["position"][1]
        asteroid_vel_x = asteroid["velocity"][0]
        asteroid_vel_y = asteroid["velocity"][1]
        
        asteroid_ship_x = ship_pos_x - asteroid_pos_x
        asteroid_ship_y = ship_pos_y - asteroid_pos_y
        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)
        asteroid_direction = math.atan2(asteroid_vel_y, asteroid_vel_x)
        
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        
        asteroid_vel = math.sqrt(asteroid_vel_x**2 + asteroid_vel_y**2)
        bullet_speed = 800
        distance = math.sqrt(asteroid_ship_x**2 + asteroid_ship_y**2)
        
        # Quadratic formula arrangement
        targ_det = (-2 * distance * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (distance**2))
        
        if targ_det < 0:
            return None, None, distance, asteroid_vel
        
        intrcpt1 = ((2 * distance * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 - bullet_speed**2))
        intrcpt2 = ((2 * distance * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2 - bullet_speed**2))
        
        # Choose positive intercept time
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
        
        # Calculate intercept point
        intrcpt_x = asteroid_pos_x + asteroid_vel_x * (bullet_t + 1/30)
        intrcpt_y = asteroid_pos_y + asteroid_vel_y * (bullet_t + 1/30)
        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y), (intrcpt_x - ship_pos_x))
        shooting_theta = my_theta1 - (math.pi/180 * ship_state["heading"])
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        return bullet_t, shooting_theta, distance, asteroid_vel
    
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
       
        return thrust, turn_rate, fire, drop_mine
    
    @property
    def name(self) -> str:
        return "Fuzzy Controller"