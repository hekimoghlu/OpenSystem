/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#if JSC_OBJC_API_ENABLED

#import "JSScript.h"
#import "SourceCode.h"
#import <wtf/RefPtr.h>

NS_ASSUME_NONNULL_BEGIN

namespace JSC {
class CachedBytecode;
class Identifier;
class JSSourceCode;
};

namespace WTF {
class String;
};

@interface JSScript(Internal)

- (instancetype)init;
- (unsigned)hash;
- (const WTF::String&)source;
- (RefPtr<JSC::CachedBytecode>)cachedBytecode;
- (JSC::JSSourceCode*)jsSourceCode;
- (JSC::SourceCode)sourceCode;
- (BOOL)writeCache:(String&)error;

@end

NS_ASSUME_NONNULL_END

#endif // JSC_OBJC_API_ENABLED
