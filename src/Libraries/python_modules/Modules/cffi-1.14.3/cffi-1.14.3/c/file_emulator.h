/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
static PyObject *PyIOBase_TypeObj;

static int init_file_emulator(void)
{
    if (PyIOBase_TypeObj == NULL) {
        PyObject *io = PyImport_ImportModule("_io");
        if (io == NULL)
            return -1;
        PyIOBase_TypeObj = PyObject_GetAttrString(io, "_IOBase");
        if (PyIOBase_TypeObj == NULL)
            return -1;
    }
    return 0;
}


#define PyFile_Check(p)  PyObject_IsInstance(p, PyIOBase_TypeObj)


static void _close_file_capsule(PyObject *ob_capsule)
{
    FILE *f = (FILE *)PyCapsule_GetPointer(ob_capsule, "FILE");
    if (f != NULL)
        fclose(f);
}


static FILE *PyFile_AsFile(PyObject *ob_file)
{
    PyObject *ob, *ob_capsule = NULL, *ob_mode = NULL;
    FILE *f;
    int fd;
    const char *mode;

    ob = PyObject_CallMethod(ob_file, "flush", NULL);
    if (ob == NULL)
        goto fail;
    Py_DECREF(ob);

    ob_capsule = PyObject_GetAttrString(ob_file, "__cffi_FILE");
    if (ob_capsule == NULL) {
        PyErr_Clear();

        fd = PyObject_AsFileDescriptor(ob_file);
        if (fd < 0)
            goto fail;

        ob_mode = PyObject_GetAttrString(ob_file, "mode");
        if (ob_mode == NULL)
            goto fail;
        mode = PyText_AsUTF8(ob_mode);
        if (mode == NULL)
            goto fail;

        fd = dup(fd);
        if (fd < 0) {
            PyErr_SetFromErrno(PyExc_OSError);
            goto fail;
        }

        f = fdopen(fd, mode);
        if (f == NULL) {
            close(fd);
            PyErr_SetFromErrno(PyExc_OSError);
            goto fail;
        }
        setbuf(f, NULL);    /* non-buffered */
        Py_DECREF(ob_mode);
        ob_mode = NULL;

        ob_capsule = PyCapsule_New(f, "FILE", _close_file_capsule);
        if (ob_capsule == NULL) {
            fclose(f);
            goto fail;
        }

        if (PyObject_SetAttrString(ob_file, "__cffi_FILE", ob_capsule) < 0)
            goto fail;
    }
    else {
        f = PyCapsule_GetPointer(ob_capsule, "FILE");
    }
    Py_DECREF(ob_capsule);   /* assumes still at least one reference */
    return f;

 fail:
    Py_XDECREF(ob_mode);
    Py_XDECREF(ob_capsule);
    return NULL;
}
