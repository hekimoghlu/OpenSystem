/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#include "config.h"
#include "sjeng.h"
#include "extvars.h"
#include "protos.h"

bool is_draw (void)
{
  /* GCP: this is straigth from Faile but converted to Sjeng internals */ 
  
  /* is_draw () is used to see if a position is a draw.  Some notes:
   *  - the 2 repetition trick is attempted first: if we have already seen a
   *    position in the search history (not game history), we haven't found
   *    anything better to do than repeat, and searching the position again would
   *    be a waste of time
   *  - if there is no match on the 2 repetition check, we look for an actual
   *    3 fold repetition
   *  - we can't check for the 50 move rule here, since the 50 move rule must be
   *    checked at the end of a search node due to mates  on the 50th move, yet
   *    we want to check for a draw by repetition before we waste any time
   *    searching the position or generating moves
   *  - is_draw () can be used in both search () and search_root () as the
   *    for loop for the 2 repetition trick will exit immediately at the root */

  int i, repeats = 0, end, start;

  /* Check on the 2 repetition trick.  Some notes:
   * - a position can't possibly have occurred once before if fifty isn't
   *   at least 4
   * - the end point is picked to look at the least number of positions, ie
   *   if going to the last capture is shorter than looking at the whole search
   *   history, we will do only that much */
  if (fifty >= 4)
    {
      if ((move_number) < (move_number + ply - 1 - fifty))
	{
	  end = move_number + ply - 1 - fifty;
	}
      else
	{
	  end = move_number;
	}
      for (i = (move_number + ply - 3); i >= 0 && i >= end; i -= 2)
	{
	  if (hash == hash_history[i])
	    {
	      return TRUE;
	    }
	}
    }

  /* Check for actual 3 repetition match.  Some notes:
   *  - a position can't possibly have occurred twice before if fifty isn't
   *    at least 6
   *  - there is no need for us to consider positions encountered during search,
   *    as if there was a match on any of them, it would have been found above
   *  - we need to adjust the starting point here based on who's turn it is:
   *    if it's the same as at the root, we need to subtract one extra */
  if (fifty >= 6)
    {
      start = move_number - 1 - (ply % 2);
      end = move_number + ply - 1 - fifty;
      
      for (i = start; i >= 0 && i >= end; i -= 2)
	{
	  if (hash == hash_history[i])
	    {
	      repeats++;
	    }
	  if (repeats >= 2)
	    {
	      return TRUE;
	    }
	}
    }

  return FALSE;

};
