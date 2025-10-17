/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
int exp_get_next_event _ANSI_ARGS_((Tcl_Interp *,ExpState **, int, ExpState **, int, int));
int exp_get_next_event_info _ANSI_ARGS_((Tcl_Interp *, ExpState *));
int exp_dsleep _ANSI_ARGS_((Tcl_Interp *, double));
void exp_init_event _ANSI_ARGS_((void));

extern void (*exp_event_exit) _ANSI_ARGS_((Tcl_Interp *));

void exp_event_disarm _ANSI_ARGS_((ExpState *,Tcl_FileProc *));
void exp_event_disarm_bg _ANSI_ARGS_((ExpState *));
void exp_event_disarm_fg _ANSI_ARGS_((ExpState *));

void exp_arm_background_channelhandler _ANSI_ARGS_((ExpState *));
void exp_disarm_background_channelhandler _ANSI_ARGS_((ExpState *));
void exp_disarm_background_channelhandler_force _ANSI_ARGS_((ExpState *));
void exp_unblock_background_channelhandler _ANSI_ARGS_((ExpState *));
void exp_block_background_channelhandler _ANSI_ARGS_((ExpState *));

void exp_background_channelhandler _ANSI_ARGS_((ClientData,int));

