import actor


class Camera(actor.Actor):
    """
    This class is nothing different from Actor class.
    But it is likely that I would like to implement special features for camera so I made this child class.
    At the moment, it doesn't have any special features.
    
    이 클래스는 Actor 클래스와 다를 것이 없습니다.
    하지만 언젠가 카메라에 특별한 기능을 추가하고 싶어지는 순간이 올 수 있기 때문에 이 클래스를 만들었습니다.
    지금 당장은 특별한 기능은 없습니다.
    """
    def __init__(self):
        super().__init__(initPos_t=[0, 1.6, 0])
