from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, P=1, I=0, D=0, target=140):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integral = 0  # 积分项初始化
        self.previous_error = 0  # 上一次误差初始化

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time')

        # BG is the only state for this PID controller
        # bg = observation.CGM
        bg = observation.BG
        error = bg - self.target  # 计算误差
        self.integral += error * sample_time  # 计算积分项
        derivative = (error - self.previous_error) / sample_time  # 计算微分项

        k = 1
        p = 0

        control_input = self.P * error  + \
                        p * self.I * self.integral + \
                        k * self.D * derivative

        if error-self.previous_error < 0:
            control_input = 0
        logger.info('Control input: {}'.format(control_input))

        self.previous_error = error

        # update the states
        # self.prev_state = bg
        # self.integrated_state += (bg - self.target) * sample_time
        logger.info('prev state: {}'.format(self.previous_error))
        logger.info('integrated state: {}'.format(self.integral))

        # return the action
        action = Action(basal=control_input, bolus=0)
        return action

    def reset(self):
        # self.integrated_state = 0
        # self.prev_state = 0
        self.integral = 0
        self.previous_error = 0
