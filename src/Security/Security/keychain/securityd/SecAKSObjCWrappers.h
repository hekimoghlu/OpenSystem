/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#import <Foundation/Foundation.h>
#import "SecKeybagSupport.h"
#include <utilities/SecCFRelease.h>

#define BridgeCFErrorToNSErrorOut(nsErrorOut, CFErr) \
{ \
    if (nsErrorOut) { \
        *nsErrorOut = CFBridgingRelease(CFErr); \
        CFErr = NULL; \
    } \
    else { \
        CFReleaseNull(CFErr); \
    } \
}

NS_ASSUME_NONNULL_BEGIN

@interface SecAKSObjCWrappers : NSObject
+ (bool)aksEncryptWithKeybag:(keybag_handle_t)keybag keyclass:(keyclass_t)keyclass plaintext:(NSData*)plaintext
                 outKeyclass:(keyclass_t* _Nullable)outKeyclass ciphertext:(NSMutableData*)ciphertext personaId:(const void* _Nullable)personaId personaIdLength:(size_t)personaIdLength error:(NSError**)error;

+ (bool)aksDecryptWithKeybag:(keybag_handle_t)keybag keyclass:(keyclass_t)keyclass ciphertext:(NSData*)ciphertext
                 outKeyclass:(keyclass_t* _Nullable)outKeyclass plaintext:(NSMutableData*)plaintext personaId:(const void* _Nullable)personaId personaIdLength:(size_t)personaIdLength error:(NSError**)error;
@end

NS_ASSUME_NONNULL_END
