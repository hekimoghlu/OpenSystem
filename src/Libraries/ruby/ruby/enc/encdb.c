/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#include "internal.h"

#define ENC_REPLICATE(name, orig) rb_encdb_replicate((name), (orig))
#define ENC_ALIAS(name, orig) rb_encdb_alias((name), (orig))
#define ENC_DUMMY(name) rb_encdb_dummy(name)
#define ENC_DEFINE(name) rb_encdb_declare(name)
#define ENC_SET_BASE(name, orig) rb_enc_set_base((name), (orig))
#define ENC_SET_DUMMY(name, orig) rb_enc_set_dummy(name)
#define ENC_DUMMY_UNICODE(name) rb_encdb_set_unicode(rb_enc_set_dummy(ENC_REPLICATE((name), name "BE")))

void
Init_encdb(void)
{
#include "encdb.h"
}
