/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#ifndef _TKUNDO
#define _TKUNDO

#ifndef _TKINT
#include "tkInt.h"
#endif

#ifdef BUILD_tk
# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLEXPORT
#endif

/*
 * Enum definining the types used in an undo stack.
 */

typedef enum {
    TK_UNDO_SEPARATOR,		/* Marker */
    TK_UNDO_ACTION		/* Command */
} TkUndoAtomType;

/*
 * Callback proc type to carry out an undo or redo action via C code. (Actions
 * can also be defined by Tcl scripts).
 */

typedef int (TkUndoProc)(Tcl_Interp *interp, ClientData clientData,
			Tcl_Obj *objPtr);

/*
 * Struct defining a single action, one or more of which may be defined (and
 * stored in a linked list) separately for each undo and redo action of an
 * undo atom.
 */

typedef struct TkUndoSubAtom {
    Tcl_Command command;	/* Tcl token used to get the current Tcl
				 * command name which will be used to execute
				 * apply/revert scripts. If NULL then it is
				 * assumed the apply/revert scripts already
				 * contain everything. */
    TkUndoProc *funcPtr;	/* Function pointer for callback to perform
				 * undo/redo actions. */
    ClientData clientData;	/* Data for 'funcPtr'. */
    Tcl_Obj *action;		/* Command to apply the action that was
				 * taken. */
    struct TkUndoSubAtom *next;	/* Pointer to the next element in the linked
				 * list. */
} TkUndoSubAtom;

/*
 * Struct representing a single undo+redo atom to be placed in the stack.
 */

typedef struct TkUndoAtom {
    TkUndoAtomType type;	/* The type that will trigger the required
				 * action. */
    TkUndoSubAtom *apply;	/* Linked list of 'apply' actions to perform
				 * for this operation. */
    TkUndoSubAtom *revert;	/* Linked list of 'revert' actions to perform
				 * for this operation. */
    struct TkUndoAtom *next;	/* Pointer to the next element in the
				 * stack. */
} TkUndoAtom;

/*
 * Struct defining a single undo+redo stack.
 */

typedef struct TkUndoRedoStack {
    TkUndoAtom *undoStack;	/* The undo stack. */
    TkUndoAtom *redoStack;	/* The redo stack. */
    Tcl_Interp *interp;		/* The interpreter in which to execute the
				 * revert and apply scripts. */
    int maxdepth;
    int depth;
} TkUndoRedoStack;

/*
 * Basic functions.
 */

MODULE_SCOPE void	TkUndoPushStack(TkUndoAtom **stack, TkUndoAtom *elem);
MODULE_SCOPE TkUndoAtom *TkUndoPopStack(TkUndoAtom **stack);
MODULE_SCOPE int	TkUndoInsertSeparator(TkUndoAtom **stack);
MODULE_SCOPE void	TkUndoClearStack(TkUndoAtom **stack);

/*
 * Functions for working on an undo/redo stack.
 */

MODULE_SCOPE TkUndoRedoStack *TkUndoInitStack(Tcl_Interp *interp, int maxdepth);
MODULE_SCOPE void	TkUndoSetDepth(TkUndoRedoStack *stack, int maxdepth);
MODULE_SCOPE void	TkUndoClearStacks(TkUndoRedoStack *stack);
MODULE_SCOPE void	TkUndoFreeStack(TkUndoRedoStack *stack);
MODULE_SCOPE void	TkUndoInsertUndoSeparator(TkUndoRedoStack *stack);
MODULE_SCOPE TkUndoSubAtom *TkUndoMakeCmdSubAtom(Tcl_Command command,
			    Tcl_Obj *actionScript, TkUndoSubAtom *subAtomList);
MODULE_SCOPE TkUndoSubAtom *TkUndoMakeSubAtom(TkUndoProc *funcPtr,
			    ClientData clientData, Tcl_Obj *actionScript,
			    TkUndoSubAtom *subAtomList);
MODULE_SCOPE void	TkUndoPushAction(TkUndoRedoStack *stack,
			    TkUndoSubAtom *apply, TkUndoSubAtom *revert);
MODULE_SCOPE int	TkUndoRevert(TkUndoRedoStack *stack);
MODULE_SCOPE int	TkUndoApply(TkUndoRedoStack *stack);

# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLIMPORT

#endif /* _TKUNDO */
