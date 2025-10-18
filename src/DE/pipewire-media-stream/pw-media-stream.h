#pragma once

#include <gtk/gtk.h>

G_BEGIN_DECLS

#define PW_TYPE_MEDIA_STREAM (pw_media_stream_get_type())
G_DECLARE_FINAL_TYPE (PwMediaStream, pw_media_stream, PW, MEDIA_STREAM, GtkMediaStream)

GtkMediaStream * pw_media_stream_new (int        pipewire_fd,
                                      uint32_t   node_id,
                                      GError   **error);

G_END_DECLS
