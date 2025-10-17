/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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

#include <wtf/RetainPtr.h>

OBJC_CLASS NSData;
OBJC_CLASS NSSet;
OBJC_CLASS NSString;
OBJC_CLASS NSURL;
OBJC_CLASS NSURLRequest;

namespace WebCore {

WEBCORE_EXPORT NSSet *QLPreviewGetSupportedMIMETypesSet();
WEBCORE_EXPORT void removeQLPreviewConverterForURL(NSURL *);
WEBCORE_EXPORT RetainPtr<NSURLRequest> registerQLPreviewConverterIfNeeded(NSURL *, NSString *mimeType, NSData *);
WEBCORE_EXPORT bool isQuickLookPreviewURL(const URL&);
WEBCORE_EXPORT NSString *createTemporaryFileForQuickLook(NSString *fileName);

static constexpr auto QLPreviewProtocol = "x-apple-ql-id"_s;

} // namespace WebCore
