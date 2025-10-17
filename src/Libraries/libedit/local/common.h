/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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
#ifndef _h_common_c
#define _h_common_c
protected el_action_t	ed_end_of_file (EditLine *, Int);
protected el_action_t	ed_insert (EditLine *, Int);
protected el_action_t	ed_delete_prev_word (EditLine *, Int);
protected el_action_t	ed_delete_next_char (EditLine *, Int);
protected el_action_t	ed_kill_line (EditLine *, Int);
protected el_action_t	ed_move_to_end (EditLine *, Int);
protected el_action_t	ed_move_to_beg (EditLine *, Int);
protected el_action_t	ed_transpose_chars (EditLine *, Int);
protected el_action_t	ed_next_char (EditLine *, Int);
protected el_action_t	ed_prev_word (EditLine *, Int);
protected el_action_t	ed_prev_char (EditLine *, Int);
protected el_action_t	ed_quoted_insert (EditLine *, Int);
protected el_action_t	ed_digit (EditLine *, Int);
protected el_action_t	ed_argument_digit (EditLine *, Int);
protected el_action_t	ed_unassigned (EditLine *, Int);
protected el_action_t	ed_tty_sigint (EditLine *, Int);
protected el_action_t	ed_tty_dsusp (EditLine *, Int);
protected el_action_t	ed_tty_flush_output (EditLine *, Int);
protected el_action_t	ed_tty_sigquit (EditLine *, Int);
protected el_action_t	ed_tty_sigtstp (EditLine *, Int);
protected el_action_t	ed_tty_stop_output (EditLine *, Int);
protected el_action_t	ed_tty_start_output (EditLine *, Int);
protected el_action_t	ed_newline (EditLine *, Int);
protected el_action_t	ed_delete_prev_char (EditLine *, Int);
protected el_action_t	ed_clear_screen (EditLine *, Int);
protected el_action_t	ed_redisplay (EditLine *, Int);
protected el_action_t	ed_start_over (EditLine *, Int);
protected el_action_t	ed_sequence_lead_in (EditLine *, Int);
protected el_action_t	ed_prev_history (EditLine *, Int);
protected el_action_t	ed_next_history (EditLine *, Int);
protected el_action_t	ed_search_prev_history (EditLine *, Int);
protected el_action_t	ed_search_next_history (EditLine *, Int);
protected el_action_t	ed_prev_line (EditLine *, Int);
protected el_action_t	ed_next_line (EditLine *, Int);
protected el_action_t	ed_command (EditLine *, Int);
#endif /* _h_common_c */
