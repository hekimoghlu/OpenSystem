/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#pragma once

#if ENABLE(ENCRYPTED_MEDIA) && USE(GSTREAMER)

#include <CDMProxy.h>
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

G_BEGIN_DECLS

#define WEBKIT_TYPE_MEDIA_CENC_DECRYPT          (webkit_media_common_encryption_decrypt_get_type())
#define WEBKIT_MEDIA_CENC_DECRYPT(obj)          (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_MEDIA_CENC_DECRYPT, WebKitMediaCommonEncryptionDecrypt))
#define WEBKIT_MEDIA_CENC_DECRYPT_CLASS(klass)  (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_MEDIA_CENC_DECRYPT, WebKitMediaCommonEncryptionDecryptClass))
#define WEBKIT_MEDIA_CENC_DECRYPT_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj), WEBKIT_TYPE_MEDIA_CENC_DECRYPT, WebKitMediaCommonEncryptionDecryptClass))

#define WEBKIT_IS_MEDIA_CENC_DECRYPT(obj)       (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_MEDIA_CENC_DECRYPT))
#define WEBKIT_IS_MEDIA_CENC_DECRYPT_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_MEDIA_CENC_DECRYPT))

typedef struct _WebKitMediaCommonEncryptionDecrypt        WebKitMediaCommonEncryptionDecrypt;
typedef struct _WebKitMediaCommonEncryptionDecryptClass   WebKitMediaCommonEncryptionDecryptClass;
typedef struct _WebKitMediaCommonEncryptionDecryptPrivate WebKitMediaCommonEncryptionDecryptPrivate;

GType webkit_media_common_encryption_decrypt_get_type(void);

bool webKitMediaCommonEncryptionDecryptIsAborting(WebKitMediaCommonEncryptionDecrypt*);

struct _WebKitMediaCommonEncryptionDecrypt {
    GstBaseTransform parent;

    WebKitMediaCommonEncryptionDecryptPrivate* priv;
};

struct _WebKitMediaCommonEncryptionDecryptClass {
    GstBaseTransformClass parentClass;

    const char* (*protectionSystemId)(WebKitMediaCommonEncryptionDecrypt*);
    bool (*cdmProxyAttached)(WebKitMediaCommonEncryptionDecrypt*, const RefPtr<WebCore::CDMProxy>&);
    bool (*decrypt)(WebKitMediaCommonEncryptionDecrypt*, GstBuffer* ivBuffer, GstBuffer* keyIDBuffer, GstBuffer* buffer, unsigned subsamplesCount, GstBuffer* subsamplesBuffer);
};

G_END_DECLS

// This function returns a C++ type. It's internal to the decryptors so it is safe to move it here to avoid the C++ return warning because of the C only linkage
// area.
WeakPtr<WebCore::CDMProxyDecryptionClient> webKitMediaCommonEncryptionDecryptGetCDMProxyDecryptionClient(WebKitMediaCommonEncryptionDecrypt*);

#endif // ENABLE(ENCRYPTED_MEDIA) && USE(GSTREAMER)
