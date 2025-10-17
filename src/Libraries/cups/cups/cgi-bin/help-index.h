/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#ifndef _CUPS_HELP_INDEX_H_
#  define _CUPS_HELP_INDEX_H_

/*
 * Include necessary headers...
 */

#  include <cups/array.h>


/*
 * C++ magic...
 */

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */

/*
 * Data structures...
 */

typedef struct help_word_s		/**** Help word structure... ****/
{
  int		count;			/* Number of occurrences */
  char		*text;			/* Word text */
} help_word_t;

typedef struct help_node_s		/**** Help node structure... ****/
{
  char		*filename;		/* Filename, relative to help dir */
  char		*section;		/* Section name (NULL if none) */
  char		*anchor;		/* Anchor name (NULL if none) */
  char		*text;			/* Text in anchor */
  cups_array_t	*words;			/* Words after this node */
  time_t	mtime;			/* Last modification time */
  off_t		offset;			/* Offset in file */
  size_t	length;			/* Length in bytes */
  int		score;			/* Search score */
} help_node_t;

typedef struct help_index_s		/**** Help index structure ****/
{
  int		search;			/* 1 = search index, 0 = normal */
  cups_array_t	*nodes;			/* Nodes sorted by filename */
  cups_array_t	*sorted;		/* Nodes sorted by score + text */
} help_index_t;


/*
 * Functions...
 */

extern void		helpDeleteIndex(help_index_t *hi);
extern help_node_t	*helpFindNode(help_index_t *hi, const char *filename,
			              const char *anchor);
extern help_index_t	*helpLoadIndex(const char *hifile, const char *directory);
extern int		helpSaveIndex(help_index_t *hi, const char *hifile);
extern help_index_t	*helpSearchIndex(help_index_t *hi, const char *query,
			                 const char *section,
					 const char *filename);


#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_CUPS_HELP_INDEX_H_ */
