"""
Module quản lý trạng thái từng ghế sử dụng kỹ thuật Hysteresis (trụ cột Bất định).
Chống tình trạng nhấp nháy chuyển màu (flickering) khi người cử động, cúi đầu hoặc bị che khuất tạm thời.
"""

class SeatState:
    """
    Theo dõi và làm mượt trạng thái một ghế.
    Chỉ đổi trạng thái khi phát hiện liên tục đủ số khung hình quy định.
    """
    def __init__(self, seat_id: int, on_frames: int = 10, off_frames: int = 25):
        """
        :param seat_id: Định danh ghế
        :param on_frames: Số frame liên tiếp nhận diện 'có người' để chuyển sang Occupied (đỏ)
        :param off_frames: Số frame liên tiếp nhận diện 'trống' để chuyển sang Empty (xanh)
        """
        self.seat_id = seat_id
        self.occupied = False
        self.count = 0
        self.on_frames = on_frames
        self.off_frames = off_frames

    def update(self, detected: bool) -> bool:
        """
        Cập nhật trạng thái thô tại khung hình hiện tại.
        :param detected: True nếu khung hiện tại có người gán vào ghế, False nếu không
        :return: Trạng thái đã làm mượt (True: có người, False: trống)
        """
        if detected != self.occupied:
            self.count += 1
            need = self.on_frames if detected else self.off_frames
            if self.count >= need:
                self.occupied = detected
                self.count = 0
        else:
            self.count = 0
        return self.occupied
