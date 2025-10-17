/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#ifndef __KEYMGR_H
#define __KEYMGR_H

#ifdef __cplusplus
extern "C" {
#endif



/*
 * keymgr - Create and maintain process-wide global data known to 
 *	    all threads across all dynamic libraries. 
 *
 */
 
enum {
  NM_ALLOW_RECURSION = 1,
  NM_RECURSION_ILLEGAL = 2
};
	
#if defined(__ppc__) || defined(__ppc64__)
extern void * _keymgr_get_per_thread_data (unsigned int key);
extern int _keymgr_set_per_thread_data (unsigned int key, void *keydata);
#endif
extern void *_keymgr_get_and_lock_processwide_ptr (unsigned int key);
extern int _keymgr_get_and_lock_processwide_ptr_2 (unsigned int key, void **);
extern int _keymgr_set_and_unlock_processwide_ptr (unsigned int key, 
						   void *ptr);
extern int _keymgr_unlock_processwide_ptr (unsigned int key);
extern int _keymgr_set_lockmode_processwide_ptr (unsigned int key, 
						 unsigned int mode);
extern unsigned int _keymgr_get_lockmode_processwide_ptr (unsigned int key);
extern int _keymgr_get_lock_count_processwide_ptr (unsigned int key);
extern void __keymgr_dwarf2_register_sections (void);

/*
 * Keys currently in use:
 */

/* Head pointer of exception context node.  */
#define KEYMGR_EH_CONTEXT_KEY		1

/* New handler.  */
#define KEYMGR_NEW_HANDLER_KEY		2

/* Unexpected exception handler.  */
#define KEYMGR_UNEXPECTED_HANDLER_KEY	3

/* Terminate handler.  */
#define KEYMGR_TERMINATE_HANDLER_KEY	4

/* Runtime mode bits.  */
#define KEYMGR_MODE_BITS		5

/* Head pointer of the list of open streams.  */
#define KEYMGR_IO_LIST			6

/* libstdc++ for GCC 2.95 stdin.  */
#define KEYMGR_IO_STDIN			7

/* libstdc++ for GCC 2.95 stdout.  */
#define KEYMGR_IO_STDOUT		8

/* libstdc++ for GCC 2.95 stdout.  */
#define KEYMGR_IO_STDERR		9

/* Number of plugins/main program currently using streams in GCC 2.95.  */
#define KEYMGR_IO_REFCNT		10

/* Flags controlling the behavior of C++ I/O.  */
#define KEYMGR_IO_MODE_BITS		11

/* Head pointer for list of per image dwarf2 unwind sections.  */
#define KEYMGR_ZOE_IMAGE_LIST		12

/* C++ runtime EH global data.  */
#define KEYMGR_EH_GLOBALS_KEY		13

/* atexit() and __cxa_atexit routine list.  */
#define KEYMGR_ATEXIT_LIST		14

/* KeyMgr 3.x is the first one supporting GCC3 stuff natively.  */
#define KEYMGR_API_MAJOR_GCC3		3
/* ... with these keys.  */
#define KEYMGR_GCC3_LIVE_IMAGE_LIST	301	/* loaded images  */
#define KEYMGR_GCC3_DW2_OBJ_LIST	302	/* Dwarf2 object list  */

/*
 * Other important data.
 */
 
/* Major revision number of the keymgr API.  */
#define KEYMGR_API_REV_MAJOR		5
/* Minor revision number of the keymgr API.  */
#define KEYMGR_API_REV_MINOR		0


/* called by libSystem_initializer */
extern void __keymgr_initializer(void);

/* for inspecting keymgr version */
extern const unsigned char keymgrVersionString[];
extern const double keymgrVersionNumber;


#ifdef __cplusplus
}
#endif

#endif /* __KEYMGR_H */
