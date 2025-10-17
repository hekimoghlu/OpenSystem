/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#include "config.h"

FILE *rcfile;
char line[STR_BUFF];

int TTSize;
int ECacheSize;
int PBSize;

int cfg_booklearn;
int cfg_razordrop;
int cfg_cutdrop;
int cfg_ksafety[15][9];
int cfg_tropism[5][7];
int havercfile;
int cfg_futprune;
int cfg_devscale;
int cfg_onerep;
int cfg_recap;
int cfg_smarteval;
int cfg_attackeval;
float cfg_scalefac;

void read_rcfile (void) 
{
  int i;
  unsigned int setc;
  
  if ((rcfile = fopen ("sjeng.rc", "r")) == NULL)
    {
      printf("No configuration file!\n");

      TTSize = 300000;
      ECacheSize = 200000;
      PBSize = 200000;
      
      cfg_devscale = 1;
      cfg_scalefac = 1.0f;
      cfg_razordrop = 1;
      cfg_cutdrop = 0;
      cfg_futprune = 1;
      cfg_smarteval = 1;
      cfg_attackeval = 0;

      havercfile = 0;

      setc =   havercfile 
	    + (cfg_devscale << 1) 
	    + (((cfg_scalefac == 1.0) ? 1 : 0) << 2)
	    + (cfg_razordrop << 3)
	    + (cfg_cutdrop << 4)
	    + (cfg_futprune << 5)
	    + (cfg_smarteval << 6)
	    + (cfg_attackeval << 7);
	    
      
      sprintf(setcode, "%u", setc);
      
      initialize_eval();
      alloc_hash();
      alloc_ecache();
      
      return;
    }

  havercfile = 1;
  
  /* read in values, possibly seperated by # commented lines */
  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &TTSize);

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &ECacheSize);

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &PBSize);

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%f", &cfg_scalefac); 

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_devscale); 

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_razordrop);

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_cutdrop);

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_booklearn);

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_futprune);

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_onerep);
    
  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_recap);
  
  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_smarteval);
  
  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  sscanf(line, "%d", &cfg_attackeval);

  fgets(line, STR_BUFF, rcfile);
  while (line[0] == '#') fgets(line, STR_BUFF, rcfile);
  
  for(i = 0; i < 5; i++)
  {
      sscanf(line, "%d %d %d %d %d %d %d", 
	  &cfg_tropism[i][0], &cfg_tropism[i][1], &cfg_tropism[i][2],&cfg_tropism[i][3],
	  &cfg_tropism[i][4], &cfg_tropism[i][5], &cfg_tropism[i][6]);
      
          do { fgets(line, STR_BUFF, rcfile);} while (line[0] == '#');
  }
  

  for(i = 0; i < 15; i++)
  {
      sscanf(line, "%d %d %d %d %d %d %d %d %d",
	  &cfg_ksafety[i][0], &cfg_ksafety[i][1],&cfg_ksafety[i][2],&cfg_ksafety[i][3],
	  &cfg_ksafety[i][4], &cfg_ksafety[i][5],&cfg_ksafety[i][6],&cfg_ksafety[i][7],
	  &cfg_ksafety[i][8]);
      
          do {fgets(line, STR_BUFF, rcfile);} while ((line[0] == '#') && !feof(rcfile));
  }

  setc =   havercfile 
            + (cfg_devscale << 1) 
	    + (((cfg_scalefac == 1.0) ? 1 : 0) << 2)
	    + (cfg_razordrop << 3)
	    + (cfg_cutdrop << 4)
	    + (cfg_futprune << 5)
	    + (cfg_smarteval << 6)
	    + (cfg_attackeval << 7);
	    
      
  sprintf(setcode, "%u", setc);

  initialize_eval();
  alloc_hash();
  alloc_ecache();
      
  return; 
  
}
