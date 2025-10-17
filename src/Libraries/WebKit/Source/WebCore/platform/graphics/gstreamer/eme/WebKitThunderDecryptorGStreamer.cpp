/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#include "WebKitThunderDecryptorGStreamer.h"

#if ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER) && USE(GSTREAMER)

#include "CDMProxyThunder.h"
#include "GStreamerCommon.h"
#include "GStreamerEMEUtilities.h"
#include <gcrypt.h>
#include <gst/base/gstbytereader.h>
#include <wtf/RunLoop.h>
#include <wtf/glib/WTFGType.h>

using namespace WebCore;

struct WebKitMediaThunderDecryptPrivate {
    RefPtr<CDMProxyThunder> cdmProxy;
};

static const char* protectionSystemId(WebKitMediaCommonEncryptionDecrypt*);
static bool cdmProxyAttached(WebKitMediaCommonEncryptionDecrypt*, const RefPtr<CDMProxy>&);
static bool decrypt(WebKitMediaCommonEncryptionDecrypt*, GstBuffer* iv, GstBuffer* keyid, GstBuffer* sample, unsigned subSamplesCount,
    GstBuffer* subSamples);

GST_DEBUG_CATEGORY(webkitMediaThunderDecryptDebugCategory);
#define GST_CAT_DEFAULT webkitMediaThunderDecryptDebugCategory

static std::array<ASCIILiteral, 11> cencEncryptionMediaTypes = { "video/mp4"_s, "audio/mp4"_s, "video/x-h264"_s, "video/x-h265"_s, "audio/mpeg"_s,
    "audio/x-eac3"_s, "audio/x-ac3"_s, "audio/x-flac"_s, "audio/x-opus"_s, "video/x-vp9"_s, "video/x-av1"_s };
static std::array<ASCIILiteral, 7> webmEncryptionMediaTypes = { "video/webm"_s, "audio/webm"_s, "video/x-vp9"_s, "video/x-av1"_s, "audio/x-opus"_s, "audio/x-vorbis"_s, "video/x-vp8"_s };

static GstStaticPadTemplate thunderSrcTemplate = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/webm; "
        "audio/webm; "
        "video/mp4; "
        "audio/mp4; "
        "audio/mpeg; "
        "audio/x-flac; "
        "audio/x-eac3; "
        "audio/x-ac3; "
        "video/x-h264; "
        "video/x-h265; "
        "video/x-vp9; video/x-vp8; "
        "video/x-av1; "
        "audio/x-opus; audio/x-vorbis"));

WEBKIT_DEFINE_TYPE(WebKitMediaThunderDecrypt, webkit_media_thunder_decrypt, WEBKIT_TYPE_MEDIA_CENC_DECRYPT)

static GRefPtr<GstCaps> createSinkPadTemplateCaps()
{
    GRefPtr<GstCaps> caps = adoptGRef(gst_caps_new_empty());

    auto& supportedKeySystems = CDMFactoryThunder::singleton().supportedKeySystems();

    if (supportedKeySystems.isEmpty()) {
        GST_WARNING("no supported key systems in Thunder, we won't be able to decrypt anything with the its decryptor");
        return caps;
    }

    for (const auto& mediaType : cencEncryptionMediaTypes) {
        gst_caps_append_structure(caps.get(), gst_structure_new("application/x-cenc", "original-media-type", G_TYPE_STRING,
            mediaType.characters(), nullptr));
    }
    for (const auto& keySystem : supportedKeySystems) {
        for (const auto& mediaType : cencEncryptionMediaTypes) {
            gst_caps_append_structure(caps.get(), gst_structure_new("application/x-cenc", "original-media-type", G_TYPE_STRING,
                mediaType.characters(), "protection-system", G_TYPE_STRING, GStreamerEMEUtilities::keySystemToUuid(keySystem), nullptr));
        }
    }

    if (supportedKeySystems.contains(GStreamerEMEUtilities::s_WidevineKeySystem) || supportedKeySystems.contains(GStreamerEMEUtilities::s_ClearKeyKeySystem)) {
        for (const auto& mediaType :  webmEncryptionMediaTypes) {
            gst_caps_append_structure(caps.get(), gst_structure_new("application/x-webm-enc", "original-media-type", G_TYPE_STRING,
                mediaType.characters(), nullptr));
        }
    }

    GST_DEBUG("sink pad template caps %" GST_PTR_FORMAT, caps.get());

    return caps;
}

static void webkit_media_thunder_decrypt_class_init(WebKitMediaThunderDecryptClass* klass)
{
    GST_DEBUG_CATEGORY_INIT(webkitMediaThunderDecryptDebugCategory, "webkitthunderdecrypt", 0, "Thunder decrypt");

    GstElementClass* elementClass = GST_ELEMENT_CLASS(klass);
    GRefPtr<GstCaps> gstSinkPadTemplateCaps = createSinkPadTemplateCaps();
    gst_element_class_add_pad_template(elementClass, gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, gstSinkPadTemplateCaps.get()));
    gst_element_class_add_pad_template(elementClass, gst_static_pad_template_get(&thunderSrcTemplate));

    gst_element_class_set_static_metadata(elementClass, "Decrypt encrypted content using Thunder", GST_ELEMENT_FACTORY_KLASS_DECRYPTOR,
        "Decrypts encrypted media using Thunder.", "Xabier RodrÃ­guez Calvar <calvaris@igalia.com>");

    WebKitMediaCommonEncryptionDecryptClass* commonClass = WEBKIT_MEDIA_CENC_DECRYPT_CLASS(klass);
    commonClass->protectionSystemId = GST_DEBUG_FUNCPTR(protectionSystemId);
    commonClass->cdmProxyAttached = GST_DEBUG_FUNCPTR(cdmProxyAttached);
    commonClass->decrypt = GST_DEBUG_FUNCPTR(decrypt);
}

static const char* protectionSystemId(WebKitMediaCommonEncryptionDecrypt* decryptor)
{
    auto* self = WEBKIT_MEDIA_THUNDER_DECRYPT(decryptor);
    ASSERT(self->priv->cdmProxy);
    return GStreamerEMEUtilities::keySystemToUuid(self->priv->cdmProxy->keySystem());
}

static bool cdmProxyAttached(WebKitMediaCommonEncryptionDecrypt* decryptor, const RefPtr<CDMProxy>& cdmProxy)
{
    auto* self = WEBKIT_MEDIA_THUNDER_DECRYPT(decryptor);
    self->priv->cdmProxy = reinterpret_cast<CDMProxyThunder*>(cdmProxy.get());
    return self->priv->cdmProxy;
}

static bool decrypt(WebKitMediaCommonEncryptionDecrypt* decryptor, GstBuffer* ivBuffer, GstBuffer* keyIDBuffer, GstBuffer* buffer, unsigned subsampleCount,
    GstBuffer* subsamplesBuffer)
{
    auto* self = WEBKIT_MEDIA_THUNDER_DECRYPT(decryptor);
    auto* priv = self->priv;

    if (!ivBuffer || !keyIDBuffer || !buffer) {
        GST_ERROR_OBJECT(self, "invalid decrypt() parameter");
        ASSERT_NOT_REACHED();
        return false;
    }

    if (subsampleCount && !subsamplesBuffer) {
        GST_ERROR_OBJECT(self, "invalid decrypt() subsamples parameter");
        ASSERT_NOT_REACHED();
        return false;
    }

    CDMProxyThunder::DecryptionContext context = { };
    context.keyIDBuffer = keyIDBuffer;
    context.ivBuffer = ivBuffer;
    context.dataBuffer = buffer;
    context.numSubsamples = subsampleCount;
    context.subsamplesBuffer = subsampleCount ? subsamplesBuffer : nullptr;
    context.cdmProxyDecryptionClient = webKitMediaCommonEncryptionDecryptGetCDMProxyDecryptionClient(decryptor);
    bool result = priv->cdmProxy->decrypt(context);

    return result;
}

#undef GST_CAT_DEFAULT

#endif // ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER) && USE(GSTREAMER)
