/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
/*
  * System library.
  */
#include <sys_defs.h>

 /*
  * Utility library.
  */
#include <name_code.h>

 /*
  * Global library.
  */
#include <mail_addr_form.h>

static const NAME_CODE addr_form_table[] = {
    "external", MA_FORM_EXTERNAL,
    "internal", MA_FORM_INTERNAL,
    "external-first", MA_FORM_EXTERNAL_FIRST,
    "internal-first", MA_FORM_INTERNAL_FIRST,
    0, -1,
};

/* mail_addr_form_from_string - symbolic mail address to internal form */

int     mail_addr_form_from_string(const char *addr_form_name)
{
    return (name_code(addr_form_table, NAME_CODE_FLAG_NONE, addr_form_name));
}

const char *mail_addr_form_to_string(int addr_form)
{
    return (str_name_code(addr_form_table, addr_form));
}
