/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#ifndef _OS_OSSERIALIZEBINARY_H
#define _OS_OSSERIALIZEBINARY_H

enum{
	kOSSerializeDictionary   = 0x01000000U,
	kOSSerializeArray        = 0x02000000U,
	kOSSerializeSet          = 0x03000000U,
	kOSSerializeNumber       = 0x04000000U,
	kOSSerializeSymbol       = 0x08000000U,
	kOSSerializeString       = 0x09000000U,
	kOSSerializeData         = 0x0a000000U,
	kOSSerializeBoolean      = 0x0b000000U,
	kOSSerializeObject       = 0x0c000000U,
	kOSSerializeTypeMask     = 0x7F000000U,
	kOSSerializeDataMask     = 0x00FFFFFFU,
	kOSSerializeEndCollecton = 0x80000000U,
};

#define kOSSerializeBinarySignature        "\323\0\0"
#define kOSSerializeIndexedBinarySignature 0x000000D4

#endif /* _OS_OSSERIALIZEBINARY_H */
