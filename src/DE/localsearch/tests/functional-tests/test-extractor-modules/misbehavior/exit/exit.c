#include <tracker-common.h>

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	/* Test that the miner can handle unexpected exit
	 * situations from the extractor (also accounts for
	 * SIGSYS, SIGSEGV, etc).
	 */
	exit (-1);
}
