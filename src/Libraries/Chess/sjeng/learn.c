/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#include "sjeng.h"
#include "protos.h"
#include "extvars.h"
#include <inttypes.h>

typedef struct 
{
  signed Depth:7;
  unsigned OnMove:1;  
  unsigned char Bestmove;
  uint32_t 		Hash;
  uint32_t 		Hold_hash;
  int32_t		Bound;
}
LearnType;

void Learn(int score, int best, int depth)
{
  int number = 0, next = 0;
  LearnType draft;
  FILE **lrnfile;

  printf("Learning score: %d  best: %d  depth:%d  hash: %X\n", score, best, depth, hash);
  
  if (Variant == Normal)
    {
      lrnfile = &lrn_standard;
    }
  else if ((Variant == Crazyhouse) || (Variant == Bughouse))
    {
      lrnfile = &lrn_zh;
    }
  else if (Variant == Suicide)
  {
      lrnfile = &lrn_suicide;
  }
  else if (Variant == Losers)
  {
      lrnfile = &lrn_losers;
  }
  else
    return;

  fseek(*lrnfile, 0, SEEK_SET);
  fread(&number, sizeof(int), 1, *lrnfile);
  fread(&next, sizeof(int), 1, *lrnfile);
  
  if (number < 50000) number++;
  
  fseek(*lrnfile, 0, SEEK_SET);
  fwrite(&number, sizeof(int), 1, *lrnfile);
  
  next++;
  if (next == 50000) next = 1;
  
  fwrite(&next, sizeof(int), 1, *lrnfile);
  
  fseek(*lrnfile, (2*sizeof(int)) + ((next-1)*sizeof(LearnType)), SEEK_SET);
  
  draft.Depth = depth;
  draft.OnMove = ToMove;
  draft.Hash = hash;
  draft.Hold_hash = hold_hash;
  draft.Bound = score;
  draft.Bestmove = best;
  
  fwrite(&draft, sizeof(draft), 1, *lrnfile);
  
  fflush(*lrnfile);
}

void LoadLearn(void)
{
  int number = 0, posloop;
  LearnType draft;
  FILE **lrnfile;
    
  if (((Variant == Crazyhouse) || (Variant == Bughouse)) && (!lrn_zh))
    return;
  else if ((Variant == Normal) && !lrn_standard)
    return;
  else if (Variant == Suicide && !lrn_suicide)
    return;
  else if (Variant == Losers && !lrn_losers)
    return;
  
  if (Variant == Normal)
    {
      lrnfile = &lrn_standard;
    }
  else if ((Variant == Crazyhouse) || (Variant == Bughouse))
    {
      lrnfile = &lrn_zh;
    }
  else if (Variant == Suicide)
    {
      lrnfile = &lrn_suicide;
    }
  else if (Variant == Losers)
  {
      lrnfile = &lrn_losers;
  }

  fseek(*lrnfile, 0, SEEK_SET);
  fread(&number, sizeof(int), 1, *lrnfile);
  fseek(*lrnfile, 2*sizeof(int), SEEK_SET);

  for (posloop = 0; posloop < number; posloop++)
    {
      fread(&draft, sizeof(LearnType), 1, *lrnfile);
      LearnStoreTT(draft.Bound, draft.Hash, draft.Hold_hash, 
		   draft.OnMove, draft.Bestmove, draft.Depth);	  
    }

  return;
}
