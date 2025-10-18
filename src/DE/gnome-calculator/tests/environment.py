# -*- coding: UTF-8 -*-

from time import sleep
from dogtail.utils import isA11yEnabled, enableA11y
if not isA11yEnabled():
    enableA11y(True)

from common_steps import App
from dogtail.config import config
import os


def before_all(context):
    """Setup gnome-calculator stuff
    Being executed before all features
    """

    try:
        # Skip dogtail actions to print to stdout
        config.logDebugToStdOut = False
        config.typingDelay = 0.2

        context.app_class = App('gnome-calculator')

    except Exception as e:
        print("Error in before_all: %s" % e.message)


def after_step(context, step):
    try:
        if step.status == 'failed' and hasattr(context, "embed"):
            # Embed screenshot if HTML report is used
            os.system("dbus-send --print-reply --session --type=method_call " +
                      "--dest='org.gnome.Shell.Screenshot' " +
                      "'/org/gnome/Shell/Screenshot' " +
                      "org.gnome.Shell.Screenshot.Screenshot " +
                      "boolean:true boolean:false string:/tmp/screenshot.png")
            context.embed('image/png', open("/tmp/screenshot.png", 'r').read())
    except Exception as e:
        print("Error in after_step: %s" % str(e))


def after_scenario(context, scenario):
    """Teardown for each scenario
    Kill gnome-calculator (in order to make this reliable we send sigkill)
    """
    try:
        # Stop gnome-calculator
        context.app_class.kill()

        # Make some pause after scenario
        sleep(1)
    except Exception as e:
        # Stupid behave simply crashes in case exception has occurred
        print("Error in after_scenario: %s" % e.message)
