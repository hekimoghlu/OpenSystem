/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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

static void
ripper_init_eventids2_table(VALUE self)
{
    VALUE h = rb_hash_new();
    rb_define_const(self, "SCANNER_EVENT_TABLE", h);
    rb_hash_aset(h, intern_sym("CHAR"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("__end__"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("backref"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("backtick"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("comma"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("comment"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("const"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("cvar"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("embdoc"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("embdoc_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("embdoc_end"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("embexpr_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("embexpr_end"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("embvar"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("float"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("gvar"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("heredoc_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("heredoc_end"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("ident"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("ignored_nl"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("imaginary"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("int"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("ivar"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("kw"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("label"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("label_end"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("lbrace"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("lbracket"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("lparen"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("nl"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("op"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("period"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("qsymbols_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("qwords_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("rational"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("rbrace"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("rbracket"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("regexp_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("regexp_end"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("rparen"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("semicolon"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("sp"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("symbeg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("symbols_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("tlambda"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("tlambeg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("tstring_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("tstring_content"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("tstring_end"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("words_beg"), INT2FIX(1));
    rb_hash_aset(h, intern_sym("words_sep"), INT2FIX(1));
}
