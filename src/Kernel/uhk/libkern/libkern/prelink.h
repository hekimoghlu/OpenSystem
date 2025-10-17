/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#ifndef _PRELINK_H_
#define _PRELINK_H_

#define kPrelinkTextSegment                "__PRELINK_TEXT"
#define kPrelinkTextSection                "__text"

#define kPrelinkDataSegment                "__PRELINK_DATA"
#define kPrelinkDataSection                "__data"

#define kPrelinkInfoSegment                "__PRELINK_INFO"
#define kPrelinkInfoSection                "__info"
#define kBuiltinInfoSection                "__kmod_info"
#define kBuiltinStartSection               "__kmod_start"

#define kReceiptInfoSegment                "__RECEIPT_INFO"
#define kAuxKCReceiptSection               "__aux_kc_receipt"

// __DATA segment
#define kBuiltinInitSection                "__kmod_init"
#define kBuiltinTermSection                "__kmod_term"

#define kPrelinkBundlePathKey              "_PrelinkBundlePath"
#define kPrelinkExecutableRelativePathKey  "_PrelinkExecutableRelativePath"
#define kPrelinkExecutableLoadKey          "_PrelinkExecutableLoadAddr"
#define kPrelinkExecutableSourceKey        "_PrelinkExecutableSourceAddr"
#define kPrelinkExecutableSizeKey          "_PrelinkExecutableSize"
#define kPrelinkInfoDictionaryKey          "_PrelinkInfoDictionary"
#define kPrelinkInterfaceUUIDKey           "_PrelinkInterfaceUUID"
#define kPrelinkKmodInfoKey                "_PrelinkKmodInfo"
#define kPrelinkLinkStateKey               "_PrelinkLinkState"
#define kPrelinkLinkStateSizeKey           "_PrelinkLinkStateSize"
#define kPrelinkLinkKASLROffsetsKey        "_PrelinkLinkKASLROffsets"
#define kPrelinkInfoKCIDKey                "_PrelinkKCID"
#define kPrelinkInfoBootKCIDKey            "_BootKCID"
#define kPrelinkInfoPageableKCIDKey        "_PageableKCID"
#define kKCBranchStubs                     "__BRANCH_STUBS"
#define kKCBranchGots                      "__BRANCH_GOTS"

#endif /* _PRELINK_H_ */
