/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#include <ft2build.h>
#include FT_INTERNAL_OBJECTS_H
#include "pshrec.h"
#include "pshalgo.h"


  /* the Postscript Hinter module structure */
  typedef struct  PS_Hinter_Module_Rec_
  {
    FT_ModuleRec          root;
    PS_HintsRec           ps_hints;

    PSH_Globals_FuncsRec  globals_funcs;
    T1_Hints_FuncsRec     t1_funcs;
    T2_Hints_FuncsRec     t2_funcs;

  } PS_Hinter_ModuleRec, *PS_Hinter_Module;


  /* finalize module */
  FT_CALLBACK_DEF( void )
  ps_hinter_done( PS_Hinter_Module  module )
  {
    module->t1_funcs.hints = NULL;
    module->t2_funcs.hints = NULL;

    ps_hints_done( &module->ps_hints );
  }


  /* initialize module, create hints recorder and the interface */
  FT_CALLBACK_DEF( FT_Error )
  ps_hinter_init( PS_Hinter_Module  module )
  {
    FT_Memory  memory = module->root.memory;
    void*      ph     = &module->ps_hints;


    ps_hints_init( &module->ps_hints, memory );

    psh_globals_funcs_init( &module->globals_funcs );

    t1_hints_funcs_init( &module->t1_funcs );
    module->t1_funcs.hints = (T1_Hints)ph;

    t2_hints_funcs_init( &module->t2_funcs );
    module->t2_funcs.hints = (T2_Hints)ph;

    return 0;
  }


  /* returns global hints interface */
  FT_CALLBACK_DEF( PSH_Globals_Funcs )
  pshinter_get_globals_funcs( FT_Module  module )
  {
    return &((PS_Hinter_Module)module)->globals_funcs;
  }


  /* return Type 1 hints interface */
  FT_CALLBACK_DEF( T1_Hints_Funcs )
  pshinter_get_t1_funcs( FT_Module  module )
  {
    return &((PS_Hinter_Module)module)->t1_funcs;
  }


  /* return Type 2 hints interface */
  FT_CALLBACK_DEF( T2_Hints_Funcs )
  pshinter_get_t2_funcs( FT_Module  module )
  {
    return &((PS_Hinter_Module)module)->t2_funcs;
  }


  FT_DEFINE_PSHINTER_INTERFACE(
    pshinter_interface,

    pshinter_get_globals_funcs,
    pshinter_get_t1_funcs,
    pshinter_get_t2_funcs
  )


  FT_DEFINE_MODULE(
    pshinter_module_class,

    0,
    sizeof ( PS_Hinter_ModuleRec ),
    "pshinter",
    0x10000L,
    0x20000L,

    &pshinter_interface,        /* module-specific interface */

    (FT_Module_Constructor)ps_hinter_init,  /* module_init   */
    (FT_Module_Destructor) ps_hinter_done,  /* module_done   */
    (FT_Module_Requester)  NULL             /* get_interface */
  )

/* END */
