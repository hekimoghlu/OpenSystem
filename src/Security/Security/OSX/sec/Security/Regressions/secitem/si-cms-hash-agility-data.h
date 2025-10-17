/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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

#ifndef si_cms_hash_agility_data_h
#define si_cms_hash_agility_data_h

#include <stdio.h>
#include <stdint.h>

/* Random data for content */
extern unsigned char content[1024];
extern size_t  content_size;

/* Random data for hash agility attribute */
extern unsigned char attribute[32];

/* Random data for hash agility V2 attribute */
extern unsigned char _attributev2[64];

/* Valid CMS message on content with hash agility attribute */
extern uint8_t valid_message[];
extern size_t valid_message_size;
/*
 * Invalid CMS message on content with hash agility attribute.
 * Only the hash agility attribute value has been changed from the valid message.
 */
extern uint8_t invalid_message[];
extern size_t invalid_message_size;

/* Valid CMS message with no hash agility attribute */
extern unsigned char valid_no_attr[];
extern size_t valid_no_attr_size;

#include "si-cms-signing-identity-p12.h"

extern unsigned char _V2_valid_message[];
extern size_t _V2_valid_message_size;

#endif /* si_cms_hash_agility_data_h */
