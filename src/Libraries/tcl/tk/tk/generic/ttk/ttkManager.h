/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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
#ifndef _TTKMANAGER
#define _TTKMANAGER

#include "ttkTheme.h"

typedef struct TtkManager_ Ttk_Manager;

/*
 * Geometry manager specification record:
 *
 * RequestedSize computes the requested size of the master window.
 *
 * PlaceSlaves sets the position and size of all managed slaves
 * by calling Ttk_PlaceSlave().
 *
 * SlaveRemoved() is called immediately before a slave is removed.
 * NB: the associated slave window may have been destroyed when this
 * routine is called.
 * 
 * SlaveRequest() is called when a slave requests a size change.
 * It should return 1 if the request should propagate, 0 otherwise.
 */
typedef struct {			/* Manager hooks */
    Tk_GeomMgr tkGeomMgr;		/* "real" Tk Geometry Manager */

    int  (*RequestedSize)(void *managerData, int *widthPtr, int *heightPtr);
    void (*PlaceSlaves)(void *managerData);
    int  (*SlaveRequest)(void *managerData, int slaveIndex, int w, int h);
    void (*SlaveRemoved)(void *managerData, int slaveIndex);
} Ttk_ManagerSpec;

/*
 * Default implementations for Tk_GeomMgr hooks:
 */
MODULE_SCOPE void Ttk_GeometryRequestProc(ClientData, Tk_Window slave);
MODULE_SCOPE void Ttk_LostSlaveProc(ClientData, Tk_Window slave);

/*
 * Public API:
 */
MODULE_SCOPE Ttk_Manager *Ttk_CreateManager(
	Ttk_ManagerSpec *, void *managerData, Tk_Window masterWindow);
MODULE_SCOPE void Ttk_DeleteManager(Ttk_Manager *);

MODULE_SCOPE void Ttk_InsertSlave(
    Ttk_Manager *, int position, Tk_Window, void *slaveData);

MODULE_SCOPE void Ttk_ForgetSlave(Ttk_Manager *, int slaveIndex);

MODULE_SCOPE void Ttk_ReorderSlave(Ttk_Manager *, int fromIndex, int toIndex);
    /* Rearrange slave positions */

MODULE_SCOPE void Ttk_PlaceSlave(
    Ttk_Manager *, int slaveIndex, int x, int y, int width, int height);
    /* Position and map the slave */

MODULE_SCOPE void Ttk_UnmapSlave(Ttk_Manager *, int slaveIndex);
    /* Unmap the slave */

MODULE_SCOPE void Ttk_ManagerSizeChanged(Ttk_Manager *);
MODULE_SCOPE void Ttk_ManagerLayoutChanged(Ttk_Manager *);
    /* Notify manager that size (resp. layout) needs to be recomputed */

/* Utilities:
 */
MODULE_SCOPE int Ttk_SlaveIndex(Ttk_Manager *, Tk_Window);
    /* Returns: index in slave array of specified window, -1 if not found */

MODULE_SCOPE int Ttk_GetSlaveIndexFromObj(
    Tcl_Interp *, Ttk_Manager *, Tcl_Obj *, int *indexPtr);

/* Accessor functions:
 */
MODULE_SCOPE int Ttk_NumberSlaves(Ttk_Manager *);
    /* Returns: number of managed slaves */

MODULE_SCOPE void *Ttk_SlaveData(Ttk_Manager *, int slaveIndex);
    /* Returns: client data associated with slave */

MODULE_SCOPE Tk_Window Ttk_SlaveWindow(Ttk_Manager *, int slaveIndex);
    /* Returns: slave window */

MODULE_SCOPE int Ttk_Maintainable(Tcl_Interp *, Tk_Window slave, Tk_Window master);
    /* Returns: 1 if master can manage slave; 0 otherwise leaving error msg */

#endif /* _TTKMANAGER */
