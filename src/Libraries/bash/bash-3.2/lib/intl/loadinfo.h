/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
#ifndef _LOADINFO_H
#define _LOADINFO_H	1

/* Declarations of locale dependent catalog lookup functions.
   Implemented in

     localealias.c    Possibly replace a locale name by another.
     explodename.c    Split a locale name into its various fields.
     l10nflist.c      Generate a list of filenames of possible message catalogs.
     finddomain.c     Find and open the relevant message catalogs.

   The main function _nl_find_domain() in finddomain.c is declared
   in gettextP.h.
 */

#ifndef PARAMS
# if __STDC__ || defined __GNUC__ || defined __SUNPRO_C || defined __cplusplus || __PROTOTYPES
#  define PARAMS(args) args
# else
#  define PARAMS(args) ()
# endif
#endif

#ifndef internal_function
# define internal_function
#endif

/* Tell the compiler when a conditional or integer expression is
   almost always true or almost always false.  */
#ifndef HAVE_BUILTIN_EXPECT
# define __builtin_expect(expr, val) (expr)
#endif

/* Separator in PATH like lists of pathnames.  */
#if defined _WIN32 || defined __WIN32__ || defined __EMX__ || defined __DJGPP__
  /* Win32, OS/2, DOS */
# define PATH_SEPARATOR ';'
#else
  /* Unix */
# define PATH_SEPARATOR ':'
#endif

/* Encoding of locale name parts.  */
#define CEN_REVISION		1
#define CEN_SPONSOR		2
#define CEN_SPECIAL		4
#define XPG_NORM_CODESET	8
#define XPG_CODESET		16
#define TERRITORY		32
#define CEN_AUDIENCE		64
#define XPG_MODIFIER		128

#define CEN_SPECIFIC	(CEN_REVISION|CEN_SPONSOR|CEN_SPECIAL|CEN_AUDIENCE)
#define XPG_SPECIFIC	(XPG_CODESET|XPG_NORM_CODESET|XPG_MODIFIER)


struct loaded_l10nfile
{
  const char *filename;
  int decided;

  const void *data;

  struct loaded_l10nfile *next;
  struct loaded_l10nfile *successor[1];
};


/* Normalize codeset name.  There is no standard for the codeset
   names.  Normalization allows the user to use any of the common
   names.  The return value is dynamically allocated and has to be
   freed by the caller.  */
extern const char *_nl_normalize_codeset PARAMS ((const char *codeset,
						  size_t name_len));

/* Lookup a locale dependent file.
   *L10NFILE_LIST denotes a pool of lookup results of locale dependent
   files of the same kind, sorted in decreasing order of ->filename.
   DIRLIST and DIRLIST_LEN are an argz list of directories in which to
   look, containing at least one directory (i.e. DIRLIST_LEN > 0).
   MASK, LANGUAGE, TERRITORY, CODESET, NORMALIZED_CODESET, MODIFIER,
   SPECIAL, SPONSOR, REVISION are the pieces of the locale name, as
   produced by _nl_explode_name().  FILENAME is the filename suffix.
   The return value is the lookup result, either found in *L10NFILE_LIST,
   or - if DO_ALLOCATE is nonzero - freshly allocated, or possibly NULL.
   If the return value is non-NULL, it is added to *L10NFILE_LIST, and
   its ->next field denotes the chaining inside *L10NFILE_LIST, and
   furthermore its ->successor[] field contains a list of other lookup
   results from which this lookup result inherits.  */
extern struct loaded_l10nfile *
_nl_make_l10nflist PARAMS ((struct loaded_l10nfile **l10nfile_list,
			    const char *dirlist, size_t dirlist_len, int mask,
			    const char *language, const char *territory,
			    const char *codeset,
			    const char *normalized_codeset,
			    const char *modifier, const char *special,
			    const char *sponsor, const char *revision,
			    const char *filename, int do_allocate));

/* Lookup the real locale name for a locale alias NAME, or NULL if
   NAME is not a locale alias (but possibly a real locale name).
   The return value is statically allocated and must not be freed.  */
extern const char *_nl_expand_alias PARAMS ((const char *name));

/* Split a locale name NAME into its pieces: language, modifier,
   territory, codeset, special, sponsor, revision.
   NAME gets destructively modified: NUL bytes are inserted here and
   there.  *LANGUAGE gets assigned NAME.  Each of *MODIFIER, *TERRITORY,
   *CODESET, *SPECIAL, *SPONSOR, *REVISION gets assigned either a
   pointer into the old NAME string, or NULL.  *NORMALIZED_CODESET
   gets assigned the expanded *CODESET, if it is different from *CODESET;
   this one is dynamically allocated and has to be freed by the caller.
   The return value is a bitmask, where each bit corresponds to one
   filled-in value:
     XPG_MODIFIER, CEN_AUDIENCE  for *MODIFIER,
     TERRITORY                   for *TERRITORY,
     XPG_CODESET                 for *CODESET,
     XPG_NORM_CODESET            for *NORMALIZED_CODESET,
     CEN_SPECIAL                 for *SPECIAL,
     CEN_SPONSOR                 for *SPONSOR,
     CEN_REVISION                for *REVISION.
 */
extern int _nl_explode_name PARAMS ((char *name, const char **language,
				     const char **modifier,
				     const char **territory,
				     const char **codeset,
				     const char **normalized_codeset,
				     const char **special,
				     const char **sponsor,
				     const char **revision));

/* Split a locale name NAME into a leading language part and all the
   rest.  Return a pointer to the first character after the language,
   i.e. to the first byte of the rest.  */
extern char *_nl_find_language PARAMS ((const char *name));

#endif	/* loadinfo.h */
