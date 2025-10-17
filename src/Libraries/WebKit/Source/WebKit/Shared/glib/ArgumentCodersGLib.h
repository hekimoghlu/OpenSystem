/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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

#include "ArgumentCoders.h"
#include <gio/gio.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _GTlsCertificate GTlsCertificate;
typedef struct _GUnixFDList GUnixFDList;
typedef struct _GVariant GVariant;

namespace IPC {

template<> struct ArgumentCoder<GRefPtr<GByteArray>> {
    static void encode(Encoder&, const GRefPtr<GByteArray>&);
    static std::optional<GRefPtr<GByteArray>> decode(Decoder&);
};

template<> struct ArgumentCoder<GRefPtr<GVariant>> {
    static void encode(Encoder&, const GRefPtr<GVariant>&);
    static std::optional<GRefPtr<GVariant>> decode(Decoder&);
};

template<> struct ArgumentCoder<GRefPtr<GTlsCertificate>> {
    static void encode(Encoder&, const GRefPtr<GTlsCertificate>&);
    static std::optional<GRefPtr<GTlsCertificate>> decode(Decoder&);
};

template<> struct ArgumentCoder<GTlsCertificateFlags> {
    static void encode(Encoder&, GTlsCertificateFlags);
    static std::optional<GTlsCertificateFlags> decode(Decoder&);
};

template<> struct ArgumentCoder<GRefPtr<GUnixFDList>> {
    static void encode(Encoder&, const GRefPtr<GUnixFDList>&);
    static std::optional<GRefPtr<GUnixFDList>> decode(Decoder&);
};

} // namespace IPC
