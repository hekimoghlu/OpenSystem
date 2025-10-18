
#include "gkd-control.h"

#include "egg/egg-secure-memory.h"

#include <pwd.h>
#include <stdlib.h>
#include <unistd.h>

EGG_SECURE_DEFINE_GLIB_GLOBALS ();

int
main (int argc, char *argv[])
{
	const char *directory;

	directory = g_getenv ("GNOME_KEYRING_CONTROL");
	g_return_val_if_fail (directory, 1);

	if (!gkd_control_quit (directory, 0))
		return 1;

	g_printerr ("success quitting daemon\n");

	return 0;
}
