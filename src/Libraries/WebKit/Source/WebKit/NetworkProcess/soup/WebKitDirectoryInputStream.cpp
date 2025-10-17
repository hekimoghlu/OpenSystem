/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"
#include "WebKitDirectoryInputStream.h"

#include "WebKitDirectoryInputStreamData.h"
#include <glib/gi18n-lib.h>
#include <wtf/StdLibExtras.h>
#include <wtf/glib/GSpanExtras.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/glib/WTFGType.h>

struct _WebKitDirectoryInputStreamPrivate {
    GRefPtr<GFileEnumerator> enumerator;
    CString uri;

    GRefPtr<GBytes> buffer;
    bool readDone;
};

WEBKIT_DEFINE_TYPE(WebKitDirectoryInputStream, webkit_directory_input_stream, G_TYPE_INPUT_STREAM)

static GBytes* webkitDirectoryInputStreamCreateHeader(WebKitDirectoryInputStream *stream)
{
    char* header = g_strdup_printf(
        "<html><head>"
        "<title>%s</title>"
        "<meta http-equiv=\"Content-Type\" content=\"text/html;\" charset=\"UTF-8\">"
        "<style>%.*s</style>"
        "<script>%.*s</script>"
        "</head>"
        "<body>"
        "<table>"
        "<thead>"
        "<th align=\"left\">%s</th><th align=\"right\">%s</th><th align=\"right\">%s</th>"
        "</thead>",
        stream->priv->uri.data(),
        static_cast<int>(WebCore::directoryUserAgentStyleSheet.size()),
        WebCore::directoryUserAgentStyleSheet.data(),
        static_cast<int>(WebCore::directoryJavaScript.size()),
        WebCore::directoryJavaScript.data(),
        _("Name"),
        _("Size"),
        _("Date Modified"));

    return g_bytes_new_with_free_func(header, strlen(header), g_free, header);
}

static GBytes* webkitDirectoryInputStreamCreateFooter(WebKitDirectoryInputStream *stream)
{
    static const char* footer = "</table></body></html>";
    return g_bytes_new_static(footer, strlen(footer));
}

static GBytes* webkitDirectoryInputStreamCreateRow(WebKitDirectoryInputStream *stream, GFileInfo* info)
{
    if (!g_file_info_get_name(info))
        return nullptr;

    const char* name = g_file_info_get_display_name(info);
    if (!name) {
        name = g_file_info_get_name(info);
        if (!g_utf8_validate(name, -1, nullptr))
            return nullptr;
    }

    GUniquePtr<char> markupName(g_markup_escape_text(name, -1));
    GUniquePtr<char> escapedName(g_uri_escape_string(name, nullptr, FALSE));
    GUniquePtr<char> path(g_build_filename(stream->priv->uri.data(), escapedName.get(), nullptr));
    GUniquePtr<char> formattedSize(g_file_info_get_file_type(info) == G_FILE_TYPE_REGULAR ? g_format_size(g_file_info_get_size(info)) : nullptr);
    GUniquePtr<char> formattedName(g_file_info_get_file_type(info) == G_FILE_TYPE_DIRECTORY ? g_strdup_printf("1.%s", path.get()) : g_strdup_printf("%s", path.get()));
#if GLIB_CHECK_VERSION (2, 61, 2)
    GRefPtr<GDateTime> modificationTime = adoptGRef(g_file_info_get_modification_date_time(info));
#else
    GTimeVal modified;
    g_file_info_get_modification_time(info, &modified);
    GRefPtr<GDateTime> modificationTime = adoptGRef(g_date_time_new_from_timeval_local(&modified));
#endif
    GUniquePtr<char> formattedTime(g_date_time_format(modificationTime.get(), "%X"));
    GUniquePtr<char> formattedDate(g_date_time_format(modificationTime.get(), "%x"));

    char* row = g_strdup_printf(
        "<tr>"
        "<td sortable-data=\"%s\"><a href=\"%s\">%s</a></td>"
        "<td align=\"right\" sortable-data=\"%" G_GOFFSET_FORMAT "\">%s</td>"
        "<td align=\"right\" sortable-data=\"%" G_GINT64_FORMAT "\">%s&ensp;%s</td>\n"
        "</tr>",
        formattedName.get(), path.get(), markupName.get(), g_file_info_get_size(info),
        formattedSize ? formattedSize.get() : "", g_date_time_to_unix(modificationTime.get()), formattedTime.get(), formattedDate.get());
    return g_bytes_new_with_free_func(row, strlen(row), g_free, row);
}

static GBytes* webkitDirectoryInputStreamReadNextFile(WebKitDirectoryInputStream* stream, GCancellable* cancellable, GError** error)
{
    GBytes* buffer = nullptr;
    do {
        GError* fileError = nullptr;
        GRefPtr<GFileInfo> info = adoptGRef(g_file_enumerator_next_file(stream->priv->enumerator.get(), cancellable, &fileError));
        if (fileError) {
            g_propagate_error(error, fileError);
            return nullptr;
        }

        if (!info && !stream->priv->readDone) {
            stream->priv->readDone = true;
            buffer = webkitDirectoryInputStreamCreateFooter(stream);
        } else if (info)
            buffer = webkitDirectoryInputStreamCreateRow(stream, info.get());
    } while (!buffer && !stream->priv->readDone);

    return buffer;
}

static gssize webkitDirectoryInputStreamRead(GInputStream* input, void* buffer, gsize count, GCancellable* cancellable, GError** error)
{
    auto* stream = WEBKIT_DIRECTORY_INPUT_STREAM(input);

    if (stream->priv->readDone)
        return 0;

    gsize totalBytesRead = 0;
    auto destinationSpan = unsafeMakeSpan(static_cast<uint8_t*>(buffer), count);
    while (totalBytesRead < count) {
        if (!stream->priv->buffer) {
            stream->priv->buffer = adoptGRef(webkitDirectoryInputStreamReadNextFile(stream, cancellable, error));
            if (!stream->priv->buffer) {
                if (totalBytesRead)
                    g_clear_error(error);
                return totalBytesRead;
            }
        }

        auto sourceSpan = span(stream->priv->buffer);
        unsigned bytesRead = std::min(sourceSpan.size(), count - totalBytesRead);
        memcpySpan(destinationSpan.subspan(totalBytesRead, bytesRead), sourceSpan.subspan(0, bytesRead));
        if (bytesRead == sourceSpan.size())
            stream->priv->buffer = nullptr;
        else
            stream->priv->buffer = adoptGRef(g_bytes_new_from_bytes(stream->priv->buffer.get(), bytesRead, sourceSpan.size() - bytesRead));
        totalBytesRead += bytesRead;
    }

    return totalBytesRead;
}

static gboolean webkitDirectoryInputStreamClose(GInputStream* input, GCancellable* cancellable, GError** error)
{
    auto* priv = WEBKIT_DIRECTORY_INPUT_STREAM(input)->priv;

    priv->buffer = nullptr;

    return g_file_enumerator_close(priv->enumerator.get(), cancellable, error);
}

static void
webkit_directory_input_stream_class_init(WebKitDirectoryInputStreamClass* klass)
{
    auto* inputStreamClass = G_INPUT_STREAM_CLASS(klass);
    inputStreamClass->read_fn = webkitDirectoryInputStreamRead;
    inputStreamClass->close_fn = webkitDirectoryInputStreamClose;
}

GRefPtr<GInputStream> webkitDirectoryInputStreamNew(GRefPtr<GFileEnumerator>&& enumerator, CString&& uri)
{
    auto* stream = WEBKIT_DIRECTORY_INPUT_STREAM(g_object_new(WEBKIT_TYPE_DIRECTORY_INPUT_STREAM, nullptr));
    stream->priv->enumerator = WTFMove(enumerator);
    stream->priv->uri = WTFMove(uri);
    stream->priv->buffer = adoptGRef(webkitDirectoryInputStreamCreateHeader(stream));

    return adoptGRef(G_INPUT_STREAM((stream)));
}
