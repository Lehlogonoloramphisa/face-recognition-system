from ..pipeline import Pipeline


class Predict:
    def __init__(self) -> None:
        pass

    def exec(self):
        step1 = ""
        step2 = ""

        pipeline = (step1 | step2)
        try:
            for _ in pipeline:
                pass
        except StopIteration:
            return
        except KeyboardInterrupt:
            return
        finally:
            print("[INFO] End of pipeline")


class PredictCamera(Pipeline):
    def __init__(self) -> None:
        pass

    def generator(self):
        pass

    def map(self, data):
        return super().map(data)


class PredictVideo(Pipeline):
    def __init__(self) -> None:
        pass

    def generator(self):
        pass

    def map(self, data):
        return super().map(data)


class PredictDeviceWebcam(Pipeline):
    def __init__(self) -> None:
        pass

    def generator(self):
        pass

    def map(self, data):
        return super().map(data)
