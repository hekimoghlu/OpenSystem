/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#ifndef	_DLF_H
#define _DLF_H
#include <iodbc.h>

#if defined(HAVE_SHL_LOAD)
#define DLDAPI_HP_SHL
#elif defined(HAVE_LIBDL)
#define DLDAPI_SVR4_DLFCN
#elif defined(HAVE_DYLD)
#define DLDAPI_MACX
#endif

#if defined(DLDAPI_SVR4_DLFCN)
#include <dlfcn.h>
#elif defined(DLDAPI_AIX_LOAD)
#include <dlfcn.h>
#elif defined(DLDAPI_VMS_IODBC) || defined(DLDAPI_MACX)
extern void *iodbc_dlopen (char * path, int mode);
extern void *iodbc_dlsym (void * hdll, char * sym);
extern char *iodbc_dlerror ();
extern int iodbc_dlclose (void * hdll);
#else
extern void *dlopen (char * path, int mode);
extern void *dlsym (void * hdll, char * sym);
extern char *dlerror ();
extern int dlclose (void * hdll);
#endif


#ifdef DLDAPI_MACX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "mach-o/dyld.h"

#define RTLD_LAZY		0x1
#define RTLD_NOW		0x2
#define RTLD_LOCAL		0x4
#define RTLD_GLOBAL		0x8
#define RTLD_NOLOAD		0x10
#define RTLD_SHARED		0x20	/* not used, the default */
#define RTLD_UNSHARED		0x40
#define RTLD_NODELETE		0x80
#define RTLD_LAZY_UNDEF		0x100


enum ofile_type
{
  OFILE_UNKNOWN,
  OFILE_FAT,
  OFILE_ARCHIVE,
  OFILE_Mach_O
};

enum byte_sex
{
  UNKNOWN_BYTE_SEX,
  BIG_ENDIAN_BYTE_SEX,
  LITTLE_ENDIAN_BYTE_SEX
};


/*
 * The structure describing an architecture flag with the string of the flag
 * name, and the cputype and cpusubtype.
 */
struct arch_flag
{
  char *name;
  cpu_type_t cputype;
  cpu_subtype_t cpusubtype;
};

/*
 * The structure used by ofile_*() routines for object files.
 */
struct ofile
{
  char *file_name;		   /* pointer to name malloc'ed by ofile_map */
  char *file_addr;		   /* pointer to vm_allocate'ed memory       */
  unsigned long file_size;	   /* size of vm_allocate'ed memory          */
  enum ofile_type file_type;	   /* type of the file                       */

  struct fat_header *fat_header;   /* If a fat file these are filled in and  */
  struct fat_arch *fat_archs;	   /*   if needed converted to host byte sex */

  /* 
   *  If this is a fat file then these are valid and filled in 
   */
  unsigned long narch;		   /* the current architecture               */
  enum ofile_type arch_type;	   /* the type of file for this arch.        */
  struct arch_flag arch_flag;	   /* the arch_flag for this arch, the name  */
				   /*   field is pointing at space malloc'ed */
				   /*   by ofile_map.                        */

  /* 
   *  If this structure is currently referencing an archive member or 
   *  an object file that is an archive member these are valid and filled in. 
   */
  unsigned long member_offset;	   /* logical offset to the member starting  */
  char *member_addr;		   /* pointer to the member contents         */
  unsigned long member_size;	   /* actual size of the member (not rounded)*/
  struct ar_hdr *member_ar_hdr;	   /* pointer to the ar_hdr for this member  */
  char *member_name;		   /* name of this member                    */
  unsigned long member_name_size;  /* size of the member name                */
  enum ofile_type member_type;	   /* the type of file for this member       */
  cpu_type_t archive_cputype;	   /* if the archive contains objects then   */
   cpu_subtype_t		   /*   these two fields reflect the object  */
   archive_cpusubtype;		   /*   at are in the archive.               */

  /* 
   *  If this structure is currently referencing a dynamic library module 
   *  these are valid and filled in. 
   */
  struct dylib_module *modtab;	   /* the module table                       */
  unsigned long nmodtab;	   /* the number of module table entries     */
  struct dylib_module		   /* pointer to the dylib_module for this   */
      *dylib_module;		   /*   module                               */
  char *dylib_module_name;	   /* the name of the module                 */

  /* 
   *  If this structure is currently referencing an object file these are
   *  valid and filled in.  The mach_header and load commands have been 
   *  converted to the host byte sex if needed 
   */
  char *object_addr;		   /* the address of the object file         */
  unsigned long object_size;	   /* the size of the object file            */
  enum byte_sex object_byte_sex;   /* the byte sex of the object file        */
  struct mach_header *mh;	   /* the mach_header of the object file     */
  struct load_command		   /* the start of the load commands         */
      *load_commands;
};


/*
 * The structure of a dlopen() handle.
 */
struct dlopen_handle
{
  dev_t dev;		/* the path's device and inode number from stat(2) */
  ino_t ino;
  int dlopen_mode;	/* current dlopen mode for this handle */
  int dlopen_count;	/* number of times dlopen() called on this handle */
  NSModule module;	/* the NSModule returned by NSLinkModule() */
  struct dlopen_handle *prev;
  struct dlopen_handle *next;
};
#endif /* DLDAPI_MACX */

#ifndef RTLD_LOCAL
#define RTLD_LOCAL	0	/* Only if not defined by dlfcn.h */
#endif
#ifndef RTLD_LAZY
#define RTLD_LAZY	1
#endif

#ifdef RTLD_NOW
#define OPL_DL_MODE	(RTLD_NOW | RTLD_LOCAL)
#else
#define OPL_DL_MODE	(RTLD_LAZY | RTLD_LOCAL)
#endif

#if defined(DLDAPI_VMS_IODBC) || defined(DLDAPI_MACX)
#define	DLL_OPEN(dll)		(void*)iodbc_dlopen((char*)(dll), OPL_DL_MODE)
#define	DLL_PROC(hdll, sym)	(void*)iodbc_dlsym((void*)(hdll), (char*)sym)
#define	DLL_ERROR()		(char*)iodbc_dlerror()
#define	DLL_CLOSE(hdll)		iodbc_dlclose((void*)(hdll))
#else
#define	DLL_OPEN(dll)		(void*)dlopen((char*)(dll), OPL_DL_MODE)
#define	DLL_PROC(hdll, sym)	(void*)dlsym((void*)(hdll), (char*)sym)
#define	DLL_ERROR()		(char*)dlerror()
#define	DLL_CLOSE(hdll)		dlclose((void*)(hdll))
#endif

#endif /* _DLF_H */
