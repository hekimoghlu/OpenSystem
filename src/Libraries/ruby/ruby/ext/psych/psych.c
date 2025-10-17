/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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

#include <psych.h>

/* call-seq: Psych.libyaml_version
 *
 * Returns the version of libyaml being used
 */
static VALUE libyaml_version(VALUE module)
{
    int major, minor, patch;
    VALUE list[3];

    yaml_get_version(&major, &minor, &patch);

    list[0] = INT2NUM((long)major);
    list[1] = INT2NUM((long)minor);
    list[2] = INT2NUM((long)patch);

    return rb_ary_new4((long)3, list);
}

VALUE mPsych;

void Init_psych(void)
{
    mPsych = rb_define_module("Psych");

    rb_define_singleton_method(mPsych, "libyaml_version", libyaml_version, 0);

    Init_psych_parser();
    Init_psych_emitter();
    Init_psych_to_ruby();
    Init_psych_yaml_tree();
}
/* vim: set noet sws=4 sw=4: */
