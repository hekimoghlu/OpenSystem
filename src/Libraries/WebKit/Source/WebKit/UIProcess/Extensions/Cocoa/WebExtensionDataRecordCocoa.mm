/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionDataRecord.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "WKWebExtensionDataRecordInternal.h"

namespace WebKit {

NSArray *WebExtensionDataRecord::errors()
{
    return [m_errors copy] ?: @[ ];
}

void WebExtensionDataRecord::addError(NSString *debugDescription, WebExtensionDataType type)
{
    if (!m_errors)
        m_errors = [[NSMutableArray alloc] init];

    switch (type) {
    case WebExtensionDataType::Local:
        [m_errors.get() addObject:createDataRecordError(WKWebExtensionDataRecordErrorLocalStorageFailed, debugDescription)];
        break;
    case WebExtensionDataType::Session:
        [m_errors.get() addObject:createDataRecordError(WKWebExtensionDataRecordErrorSessionStorageFailed, debugDescription)];
        break;
    case WebExtensionDataType::Sync:
        [m_errors.get() addObject:createDataRecordError(WKWebExtensionDataRecordErrorSynchronizedStorageFailed, debugDescription)];
        break;
    default:
        ASSERT_NOT_REACHED();
    }
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
