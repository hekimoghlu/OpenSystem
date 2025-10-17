/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#ifndef _h_emacs_c
#define _h_emacs_c
protected el_action_t	em_delete_or_list (EditLine *, Int);
protected el_action_t	em_delete_next_word (EditLine *, Int);
protected el_action_t	em_yank (EditLine *, Int);
protected el_action_t	em_kill_line (EditLine *, Int);
protected el_action_t	em_kill_region (EditLine *, Int);
protected el_action_t	em_copy_region (EditLine *, Int);
protected el_action_t	em_gosmacs_transpose (EditLine *, Int);
protected el_action_t	em_next_word (EditLine *, Int);
protected el_action_t	em_upper_case (EditLine *, Int);
protected el_action_t	em_capitol_case (EditLine *, Int);
protected el_action_t	em_lower_case (EditLine *, Int);
protected el_action_t	em_set_mark (EditLine *, Int);
protected el_action_t	em_exchange_mark (EditLine *, Int);
protected el_action_t	em_universal_argument (EditLine *, Int);
protected el_action_t	em_meta_next (EditLine *, Int);
protected el_action_t	em_toggle_overwrite (EditLine *, Int);
protected el_action_t	em_copy_prev_word (EditLine *, Int);
protected el_action_t	em_inc_search_next (EditLine *, Int);
protected el_action_t	em_inc_search_prev (EditLine *, Int);
protected el_action_t	em_delete_prev_char (EditLine *, Int);
#endif /* _h_emacs_c */
