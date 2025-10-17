/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
#ifndef _h_vi_c
#define _h_vi_c
protected el_action_t	vi_paste_next (EditLine *, Int);
protected el_action_t	vi_paste_prev (EditLine *, Int);
protected el_action_t	vi_prev_big_word (EditLine *, Int);
protected el_action_t	vi_prev_word (EditLine *, Int);
protected el_action_t	vi_next_big_word (EditLine *, Int);
protected el_action_t	vi_next_word (EditLine *, Int);
protected el_action_t	vi_change_case (EditLine *, Int);
protected el_action_t	vi_change_meta (EditLine *, Int);
protected el_action_t	vi_insert_at_bol (EditLine *, Int);
protected el_action_t	vi_replace_char (EditLine *, Int);
protected el_action_t	vi_replace_mode (EditLine *, Int);
protected el_action_t	vi_substitute_char (EditLine *, Int);
protected el_action_t	vi_substitute_line (EditLine *, Int);
protected el_action_t	vi_change_to_eol (EditLine *, Int);
protected el_action_t	vi_insert (EditLine *, Int);
protected el_action_t	vi_add (EditLine *, Int);
protected el_action_t	vi_add_at_eol (EditLine *, Int);
protected el_action_t	vi_delete_meta (EditLine *, Int);
protected el_action_t	vi_end_big_word (EditLine *, Int);
protected el_action_t	vi_end_word (EditLine *, Int);
protected el_action_t	vi_undo (EditLine *, Int);
protected el_action_t	vi_command_mode (EditLine *, Int);
protected el_action_t	vi_zero (EditLine *, Int);
protected el_action_t	vi_delete_prev_char (EditLine *, Int);
protected el_action_t	vi_list_or_eof (EditLine *, Int);
protected el_action_t	vi_kill_line_prev (EditLine *, Int);
protected el_action_t	vi_search_prev (EditLine *, Int);
protected el_action_t	vi_search_next (EditLine *, Int);
protected el_action_t	vi_repeat_search_next (EditLine *, Int);
protected el_action_t	vi_repeat_search_prev (EditLine *, Int);
protected el_action_t	vi_next_char (EditLine *, Int);
protected el_action_t	vi_prev_char (EditLine *, Int);
protected el_action_t	vi_to_next_char (EditLine *, Int);
protected el_action_t	vi_to_prev_char (EditLine *, Int);
protected el_action_t	vi_repeat_next_char (EditLine *, Int);
protected el_action_t	vi_repeat_prev_char (EditLine *, Int);
protected el_action_t	vi_match (EditLine *, Int);
protected el_action_t	vi_undo_line (EditLine *, Int);
protected el_action_t	vi_to_column (EditLine *, Int);
protected el_action_t	vi_yank_end (EditLine *, Int);
protected el_action_t	vi_yank (EditLine *, Int);
protected el_action_t	vi_comment_out (EditLine *, Int);
protected el_action_t	vi_alias (EditLine *, Int);
protected el_action_t	vi_to_history_line (EditLine *, Int);
protected el_action_t	vi_histedit (EditLine *, Int);
protected el_action_t	vi_history_word (EditLine *, Int);
protected el_action_t	vi_redo (EditLine *, Int);
#endif /* _h_vi_c */
