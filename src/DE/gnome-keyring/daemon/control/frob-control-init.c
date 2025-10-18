
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
	const gchar *env[] = { NULL };
	gchar **envp, **e;

	directory = g_getenv ("GNOME_KEYRING_CONTROL");
	g_return_val_if_fail (directory, 1);

	envp = gkd_control_initialize (directory, "pkcs11,ssh,secret", env);
	if (envp == NULL)
		return 1;

	for (e = envp; *e; ++e)
		g_printerr ("%s\n", *e);
	g_strfreev (envp);

	return 0;
}
