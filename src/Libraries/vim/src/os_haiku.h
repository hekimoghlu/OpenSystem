/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
 * os_haiku.h
 */

#define USE_TERM_CONSOLE

#define USR_VIM_DIR "$BE_USER_SETTINGS/vim"

#define USR_EXRC_FILE	USR_VIM_DIR "/exrc"
#define USR_EXRC_FILE2	USR_VIM_DIR "/vim/exrc"
#define USR_VIMRC_FILE	USR_VIM_DIR "/vimrc"
#define USR_VIMRC_FILE2	USR_VIM_DIR "/vim/vimrc"
#define USR_GVIMRC_FILE	USR_VIM_DIR "/gvimrc"
#define USR_GVIMRC_FILE2	USR_VIM_DIR "/vim/gvimrc"
#define VIMINFO_FILE	USR_VIM_DIR "/viminfo"

#ifdef RUNTIME_GLOBAL
# ifdef RUNTIME_GLOBAL_AFTER
#  define DFLT_RUNTIMEPATH	USR_VIM_DIR "," RUNTIME_GLOBAL ",$VIMRUNTIME," RUNTIME_GLOBAL_AFTER "," USR_VIM_DIR "/after"
#  define CLEAN_RUNTIMEPATH	RUNTIME_GLOBAL ",$VIMRUNTIME," RUNTIME_GLOBAL_AFTER
# else
#  define DFLT_RUNTIMEPATH	USR_VIM_DIR "," RUNTIME_GLOBAL ",$VIMRUNTIME," RUNTIME_GLOBAL "/after," USR_VIM_DIR "/after"
#  define CLEAN_RUNTIMEPATH	RUNTIME_GLOBAL ",$VIMRUNTIME," RUNTIME_GLOBAL "/after"
# endif
#else
# define DFLT_RUNTIMEPATH	USR_VIM_DIR ",$VIM/vimfiles,$VIMRUNTIME,$VIM/vimfiles/after," USR_VIM_DIR "/after"
# define CLEAN_RUNTIMEPATH	"$VIM/vimfiles,$VIMRUNTIME,$VIM/vimfiles/after"
#endif
