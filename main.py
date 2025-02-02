from tello_application import TelloApplication

if __name__ == "__main__":
    """
    Entry point for the application.
    Initializes and runs the TelloApplication, which manages
    all threads (video capture, tracking, control, status display).
    """
    app = TelloApplication()
    app.run()
