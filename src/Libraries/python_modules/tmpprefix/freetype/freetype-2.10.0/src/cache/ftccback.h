/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#ifndef FTCCBACK_H_
#define FTCCBACK_H_

#include <ft2build.h>
#include FT_CACHE_H
#include "ftcmru.h"
#include "ftcimage.h"
#include "ftcmanag.h"
#include "ftcglyph.h"
#include "ftcsbits.h"


  FT_LOCAL( void )
  ftc_inode_free( FTC_Node   inode,
                  FTC_Cache  cache );

  FT_LOCAL( FT_Error )
  ftc_inode_new( FTC_Node   *pinode,
                 FT_Pointer  gquery,
                 FTC_Cache   cache );

  FT_LOCAL( FT_Offset )
  ftc_inode_weight( FTC_Node   inode,
                    FTC_Cache  cache );


  FT_LOCAL( void )
  ftc_snode_free( FTC_Node   snode,
                  FTC_Cache  cache );

  FT_LOCAL( FT_Error )
  ftc_snode_new( FTC_Node   *psnode,
                 FT_Pointer  gquery,
                 FTC_Cache   cache );

  FT_LOCAL( FT_Offset )
  ftc_snode_weight( FTC_Node   snode,
                    FTC_Cache  cache );

  FT_LOCAL( FT_Bool )
  ftc_snode_compare( FTC_Node    snode,
                     FT_Pointer  gquery,
                     FTC_Cache   cache,
                     FT_Bool*    list_changed );


  FT_LOCAL( FT_Bool )
  ftc_gnode_compare( FTC_Node    gnode,
                     FT_Pointer  gquery,
                     FTC_Cache   cache,
                     FT_Bool*    list_changed );


  FT_LOCAL( FT_Error )
  ftc_gcache_init( FTC_Cache  cache );

  FT_LOCAL( void )
  ftc_gcache_done( FTC_Cache  cache );


  FT_LOCAL( FT_Error )
  ftc_cache_init( FTC_Cache  cache );

  FT_LOCAL( void )
  ftc_cache_done( FTC_Cache  cache );

  FT_LOCAL( void )
  ftc_node_destroy( FTC_Node     node,
                    FTC_Manager  manager );


#endif /* FTCCBACK_H_ */


/* END */
