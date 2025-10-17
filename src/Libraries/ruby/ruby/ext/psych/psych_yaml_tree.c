/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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

VALUE cPsychVisitorsYamlTree;

/*
 * call-seq: private_iv_get(target, prop)
 *
 * Get the private instance variable +prop+ from +target+
 */
static VALUE private_iv_get(VALUE self, VALUE target, VALUE prop)
{
    return rb_attr_get(target, rb_intern(StringValueCStr(prop)));
}

void Init_psych_yaml_tree(void)
{
    VALUE psych     = rb_define_module("Psych");
    VALUE visitors  = rb_define_module_under(psych, "Visitors");
    VALUE visitor   = rb_define_class_under(visitors, "Visitor", rb_cObject);
    cPsychVisitorsYamlTree = rb_define_class_under(visitors, "YAMLTree", visitor);

    rb_define_private_method(cPsychVisitorsYamlTree, "private_iv_get", private_iv_get, 2);
}
/* vim: set noet sws=4 sw=4: */
