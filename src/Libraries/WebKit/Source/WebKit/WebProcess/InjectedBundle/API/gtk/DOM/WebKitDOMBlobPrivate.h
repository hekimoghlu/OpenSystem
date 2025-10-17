/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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
#ifndef WebKitDOMBlobPrivate_h
#define WebKitDOMBlobPrivate_h

#include <WebCore/Blob.h>
#include <webkitdom/WebKitDOMBlob.h>

namespace WebKit {
WebKitDOMBlob* wrapBlob(WebCore::Blob*);
WebKitDOMBlob* kit(WebCore::Blob*);
WebCore::Blob* core(WebKitDOMBlob*);
} // namespace WebKit

#endif /* WebKitDOMBlobPrivate_h */
