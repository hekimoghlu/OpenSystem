/* This is the source code of our Coverity model. Meant to enhance
 * the accuracy of reports, improve understanding of the existing
 * code, or avoid false positives.
 *
 * This is not a real C file, and can not use #include directives.
 * These are basically hints about the function behavior, in a way
 * that Coverity can understand.
 *
 * The model file must be uploaded by someone with permissions at the
 * LocalSearch console at https://scan.coverity.com/projects/tracker-miners
 */

#define NULL ((void *) 0)

typedef long ssize_t;
typedef struct _GMappedFile { char *data } GMappedFile;
typedef unsigned int gboolean;

void
g_atomic_int_inc (int *atomic)
{
	(*atomic)++;
}

gboolean
g_atomic_int_dec_and_test (int *atomic)
{
	(*atomic)--;
	if (*atomic < 0)
		return 1;
	else
		return 0;
}

char *
g_mapped_file_get_contents (GMappedFile *mapped_file)
{
	return mapped_file->data;
}

char *
g_strchug (char *string)
{
	__coverity_string_null_sink__ (string);
	return string;
}

char *
g_strchomp (char *string)
{
	__coverity_string_null_sink__ (string);
	return string;
}

gboolean
g_utf8_validate (char *string, ssize_t size, const char **end)
{
	int is_ok;

	if (is_ok) {
		__coverity_tainted_string_sanitize_content__ (string);
		return 1;
	} else {
		return 0;
	}
}
