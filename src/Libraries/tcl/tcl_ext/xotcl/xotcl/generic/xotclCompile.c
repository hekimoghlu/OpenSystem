/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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

#include "xotclInt.h"

#ifdef XOTCL_BYTECODE
#include <tclCompile.h>

static CompileProc 
  initProcNsCompile, nextCompile, 
  selfCompile, selfDispatchCompile;

static InstructionDesc instructionTable[] = {
  {"initProc",		  1,   0,   {OPERAND_NONE}},
  {"next",		  1,   0,   {OPERAND_NONE}},
  {"self",		  1,   0,   {OPERAND_NONE}},
  {"dispatch",		  2,   1,   {OPERAND_UINT1}},
};

static XOTclCompEnv instructions[] = {
  {0, 0, initProcNsCompile, XOTclInitProcNSCmd},
  {0, 0, nextCompile, XOTclNextObjCmd},
  {0, 0, selfCompile, XOTclGetSelfObjCmd},
  {0, 0, selfDispatchCompile, /*XOTclSelfDispatchCmd*/XOTclDirectSelfDispatch},
  0
};

XOTclCompEnv *
XOTclGetCompEnv() {
  return &instructions[0];
}


static int
initProcNsCompile(Tcl_Interp *interp, Tcl_Parse *parsePtr,
		  CompileEnv *envPtr) {

  if (parsePtr->numWords != 1) {
	Tcl_ResetResult(interp);
	Tcl_AppendToObj(Tcl_GetObjResult(interp),
	        "wrong # args: should be '::xotcl::initProcNS'", -1);
	envPtr->maxStackDepth = 0;
	return TCL_ERROR;
    }

  TclEmitOpcode(instructions[INST_INITPROC].bytecode, envPtr);
  envPtr->maxStackDepth = 0;
    
  return TCL_OK;
}

static int
nextCompile(Tcl_Interp *interp, Tcl_Parse *parsePtr,
		  CompileEnv *envPtr) {

  if (parsePtr->numWords != 1) 
    return TCL_OUT_LINE_COMPILE;
    
  TclEmitOpcode(instructions[INST_NEXT].bytecode, envPtr);
  envPtr->maxStackDepth = 0;
    
  return TCL_OK;
}
static int
selfCompile(Tcl_Interp *interp, Tcl_Parse *parsePtr,
		  CompileEnv *envPtr) {

  if (parsePtr->numWords != 1) 
    return TCL_OUT_LINE_COMPILE;
    
  TclEmitOpcode(instructions[INST_SELF].bytecode, envPtr);
  envPtr->maxStackDepth = 0;
    
  return TCL_OK;
}
static int
selfDispatchCompile(Tcl_Interp *interp, Tcl_Parse *parsePtr,
		  CompileEnv *envPtr) {

  Tcl_Token *tokenPtr;
  int code, wordIdx;
  /*
  fprintf(stderr, "****** selfDispatchCompile words=%d tokens=%d, avail=%d\n",
	  parsePtr->numWords,parsePtr->numTokens,parsePtr->tokensAvailable);
  */

  if (parsePtr->numWords > 255) 
    return TCL_OUT_LINE_COMPILE;

  /*TclEmitOpcode(instructions[INST_SELF].bytecode, envPtr);*/

  for (wordIdx=0, tokenPtr = parsePtr->tokenPtr + 0; 
       wordIdx < parsePtr->numWords; 
       wordIdx++, tokenPtr += (tokenPtr->numComponents + 1)) {

    /*
    fprintf(stderr,"  %d: %p token type=%d size=%d\n",
	    wordIdx,tokenPtr,tokenPtr->type,tokenPtr->size );
    */
    if (tokenPtr->type == TCL_TOKEN_SIMPLE_WORD) {
      TclEmitPush(TclRegisterLiteral(envPtr, tokenPtr->start,
				     tokenPtr->size, 0), envPtr);
      envPtr->maxStackDepth = 1;
      /*
      fprintf(stderr,"  %d: simple '%s' components=%d\n",
	      wordIdx,tokenPtr->start, tokenPtr->numComponents);
      */
    } else {
      /*
      fprintf(stderr,"  %d NOT simple '%s' components=%d\n",
	      wordIdx,tokenPtr->start, tokenPtr->numComponents);
      */
      code = TclCompileTokens(interp, tokenPtr+1,
			      tokenPtr->numComponents, envPtr);
      if (code != TCL_OK) {
	return code;
      }
    }
  }

  /*fprintf(stderr, "maxdepth=%d, onStack=%d\n",envPtr->maxStackDepth,wordIdx);
   */
  TclEmitInstInt1(instructions[INST_SELF_DISPATCH].bytecode, wordIdx, envPtr);
  envPtr->maxStackDepth = 0;
  
  return TCL_OK;
}



void
XOTclBytecodeInit() {
  int i;
  for(i=0; i<LAST_INSTRUCTION; i++) {
    if ((instructions[i].bytecode = 
       TclRegisterUserOpcode(&instructionTable[i],
			     instructions[i].callProc,
			     instructions[i].cmdPtr->objClientData))) {
      instructions[i].cmdPtr->compileProc = instructions[i].compileProc;
    }
    
  }
  /*tclTraceCompile = 2;*/

}

#endif
