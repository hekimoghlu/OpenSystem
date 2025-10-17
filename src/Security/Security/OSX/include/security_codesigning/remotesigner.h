/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#import <CoreFoundation/CoreFoundation.h>

#import "codedirectory.h"
#import "SecCodeSignerRemote.h"


namespace Security {
namespace CodeSigning {

CF_ASSUME_NONNULL_BEGIN

/// Performs a signing operation for the provided code directory and context, using the
/// provided certificate chain and handler to get the actual signature.
/// Output is the full CMS blob that can be placed into a code signature slot.
OSStatus
doRemoteSigning(const CodeDirectory *cd,
				CFDictionaryRef hashDict,
				CFArrayRef hashList,
				CFAbsoluteTime signingTime,
				CFArrayRef certificateChain,
				SecCodeRemoteSignHandler signHandler,
				CFDataRef * _Nonnull CF_RETURNS_RETAINED outputCMS);

CF_ASSUME_NONNULL_END

}
}
