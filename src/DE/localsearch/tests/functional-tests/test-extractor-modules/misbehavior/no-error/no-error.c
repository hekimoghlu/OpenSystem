#include <tracker-common.h>

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	TrackerResource *resource;

	resource = tracker_resource_new ("fail://");
	tracker_resource_add_uri (resource, "rdf:type", "rdfs:Resource");
	tracker_extract_info_set_resource (info, resource);

	/* Return FALSE without error */
	return FALSE;
}
