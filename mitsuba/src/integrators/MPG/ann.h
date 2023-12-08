#pragma once

#include <cstring>
#include <enoki/morton.h>
#include <enoki/stl.h>
#include <fstream>
#include <iomanip>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <mutex>
#include <random>
#include <sstream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <thread>
#include <vector>

class Point3f;
class Vector3f;

NAMESPACE_BEGIN(mitsuba)

using std::cerr;
using std::cout;
using std::istream;
using std::ostream;

// a modified version of ANN

//----------------------------------------------------------------------
// File:			ANN.h
// Programmer:		Sunil Arya and David Mount
// Description:		Basic include file for approximate nearest
//					neighbor searching.
// Last modified:	01/27/10 (Version 1.1.2)
//----------------------------------------------------------------------
// Copyright (c) 1997-2010 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.0  04/01/05
//		Added copyright and revision information
//		Added ANNcoordPrec for coordinate precision.
//		Added methods theDim, nPoints, maxPoints, thePoints to ANNpointSet.
//		Cleaned up C++ structure for modern compilers
//	Revision 1.1  05/03/05
//		Added fixed-radius k-NN searching
//	Revision 1.1.2  01/27/10
//		Fixed minor compilation bugs for new versions of gcc
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// ANN - approximate nearest neighbor searching
//	ANN is a library for approximate nearest neighbor searching,
//	based on the use of standard and priority search in kd-trees
//	and balanced box-decomposition (bbd) trees. Here are some
//	references to the main algorithmic techniques used here:
//
//		kd-trees:
//			Friedman, Bentley, and Finkel, ``An algorithm for finding
//				best matches in logarithmic expected time,'' ACM
//				Transactions on Mathematical Software, 3(3):209-226, 1977.
//
//		Priority search in kd-trees:
//			Arya and Mount, ``Algorithms for fast vector quantization,''
//				Proc. of DCC '93: Data Compression Conference, eds. J. A.
//				Storer and M. Cohn, IEEE Press, 1993, 381-390.
//
//		Approximate nearest neighbor search and bbd-trees:
//			Arya, Mount, Netanyahu, Silverman, and Wu, ``An optimal
//				algorithm for approximate nearest neighbor searching,''
//				5th Ann. ACM-SIAM Symposium on Discrete Algorithms,
//				1994, 573-582.
//----------------------------------------------------------------------

#ifndef ANN_H
#define ANN_H

// #ifdef WIN32
//   //----------------------------------------------------------------------
//   // For Microsoft Visual C++, externally accessible symbols must be
//   // explicitly indicated with DLL_API, which is somewhat like "extern."
//   //
//   // The following ifdef block is the standard way of creating macros
//   // which make exporting from a DLL simpler. All files within this DLL
//   // are compiled with the DLL_EXPORTS preprocessor symbol defined on the
//   // command line. In contrast, projects that use (or import) the DLL
//   // objects do not define the DLL_EXPORTS symbol. This way any other
//   // project whose source files include this file see DLL_API functions as
//   // being imported from a DLL, wheras this DLL sees symbols defined with
//   // this macro as being exported.
//   //----------------------------------------------------------------------
//   #ifdef DLL_EXPORTS
// 	 #define DLL_API __declspec(dllexport)
//   #else
// 	#define DLL_API __declspec(dllimport)
//   #endif
//   //----------------------------------------------------------------------
//   // DLL_API is ignored for all other systems
//   //----------------------------------------------------------------------
// #else
//   #define DLL_API
// #endif

#define DLL_API

//----------------------------------------------------------------------
//  basic includes
//----------------------------------------------------------------------

#include <cmath>    // math includes
#include <cstdlib>  // standard lib includes
#include <cstring>  // C-style strings
#include <iostream> // I/O streams

//----------------------------------------------------------------------
// Limits
// There are a number of places where we use the maximum double value as
// default initializers (and others may be used, depending on the
// data/distance representation). These can usually be found in limits.h
// (as LONG_MAX, INT_MAX) or in float.h (as DBL_MAX, FLT_MAX).
//
// Not all systems have these files.  If you are using such a system,
// you should set the preprocessor symbol ANN_NO_LIMITS_H when
// compiling, and modify the statements below to generate the
// appropriate value. For practical purposes, this does not need to be
// the maximum double value. It is sufficient that it be at least as
// large than the maximum squared distance between between any two
// points.
//----------------------------------------------------------------------
#ifdef ANN_NO_LIMITS_H                // limits.h unavailable
#include <cvalues>                    // replacement for limits.h
const double ANN_DBL_MAX = MAXDOUBLE; // insert maximum double
#else
#include <cfloat>
#include <climits>
const double ANN_DBL_MAX = DBL_MAX;
#endif

#define ANNversion "1.1.2" // ANN version and information
#define ANNversionCmt ""
#define ANNcopyright "David M. Mount and Sunil Arya"
#define ANNlatestRev "Jan 27, 2010"

//----------------------------------------------------------------------
//	ANNbool
//	This is a simple boolean type. Although ANSI C++ is supposed
//	to support the type bool, some compilers do not have it.
//----------------------------------------------------------------------

enum ANNbool { ANNfalse = 0, ANNtrue = 1 }; // ANN boolean type (non ANSI C++)

//----------------------------------------------------------------------
//	ANNcoord, ANNdist
//		ANNcoord and ANNdist are the types used for representing
//		point coordinates and distances.  They can be modified by the
//		user, with some care.  It is assumed that they are both numeric
//		types, and that ANNdist is generally of an equal or higher type
//		from ANNcoord.	A variable of type ANNdist should be large
//		enough to store the sum of squared components of a variable
//		of type ANNcoord for the number of dimensions needed in the
//		application.  For example, the following combinations are
//		legal:
//
//		ANNcoord		ANNdist
//		---------		-------------------------------
//		short			short, int, long, float, double
//		int				int, long, float, double
//		long			long, float, double
//		float			float, double
//		double			double
//
//		It is the user's responsibility to make sure that overflow does
//		not occur in distance calculation.
//----------------------------------------------------------------------

typedef double ANNcoord; // coordinate data type
typedef double ANNdist;  // distance data type

//----------------------------------------------------------------------
//	ANNidx
//		ANNidx is a point index.  When the data structure is built, the
//		points are given as an array.  Nearest neighbor results are
//		returned as an integer index into this array.  To make it
//		clearer when this is happening, we define the integer type
//		ANNidx.	 Indexing starts from 0.
//
//		For fixed-radius near neighbor searching, it is possible that
//		there are not k nearest neighbors within the search radius.  To
//		indicate this, the algorithm returns ANN_NULL_IDX as its result.
//		It should be distinguishable from any valid array index.
//----------------------------------------------------------------------

typedef int ANNidx;             // point index
const ANNidx ANN_NULL_IDX = -1; // a NULL point index

//----------------------------------------------------------------------
//	Infinite distance:
//		The code assumes that there is an "infinite distance" which it
//		uses to initialize distances before performing nearest neighbor
//		searches.  It should be as larger or larger than any legitimate
//		nearest neighbor distance.
//
//		On most systems, these should be found in the standard include
//		file <limits.h> or possibly <float.h>.  If you do not have these
//		file, some suggested values are listed below, assuming 64-bit
//		long, 32-bit int and 16-bit short.
//
//		ANNdist ANN_DIST_INF	Values (see <limits.h> or <float.h>)
//		------- ------------	------------------------------------
//		double	DBL_MAX			1.79769313486231570e+308
//		float	FLT_MAX			3.40282346638528860e+38
//		long	LONG_MAX		0x7fffffffffffffff
//		int		INT_MAX			0x7fffffff
//		short	SHRT_MAX		0x7fff
//----------------------------------------------------------------------

const ANNdist ANN_DIST_INF = ANN_DBL_MAX;

//----------------------------------------------------------------------
//	Significant digits for tree dumps:
//		When floating point coordinates are used, the routine that dumps
//		a tree needs to know roughly how many significant digits there
//		are in a ANNcoord, so it can output points to full precision.
//		This is defined to be ANNcoordPrec.  On most systems these
//		values can be found in the standard include files <limits.h> or
//		<float.h>.  For integer types, the value is essentially ignored.
//
//		ANNcoord ANNcoordPrec	Values (see <limits.h> or <float.h>)
//		-------- ------------	------------------------------------
//		double	 DBL_DIG		15
//		float	 FLT_DIG		6
//		long	 doesn't matter 19
//		int		 doesn't matter 10
//		short	 doesn't matter 5
//----------------------------------------------------------------------

#ifdef DBL_DIG // number of sig. bits in ANNcoord
const int ANNcoordPrec = DBL_DIG;
#else
const int ANNcoordPrec   = 15; // default precision
#endif

//----------------------------------------------------------------------
// Self match?
//	In some applications, the nearest neighbor of a point is not
//	allowed to be the point itself. This occurs, for example, when
//	computing all nearest neighbors in a set.  By setting the
//	parameter ANN_ALLOW_SELF_MATCH to ANNfalse, the nearest neighbor
//	is the closest point whose distance from the query point is
//	strictly positive.
//----------------------------------------------------------------------

const ANNbool ANN_ALLOW_SELF_MATCH = ANNtrue;

//----------------------------------------------------------------------
//	Norms and metrics:
//		ANN supports any Minkowski norm for defining distance.  In
//		particular, for any p >= 1, the L_p Minkowski norm defines the
//		length of a d-vector (v0, v1, ..., v(d-1)) to be
//
//				(|v0|^p + |v1|^p + ... + |v(d-1)|^p)^(1/p),
//
//		(where ^ denotes exponentiation, and |.| denotes absolute
//		value).  The distance between two points is defined to be the
//		norm of the vector joining them.  Some common distance metrics
//		include
//
//				Euclidean metric		p = 2
//				Manhattan metric		p = 1
//				Max metric				p = infinity
//
//		In the case of the max metric, the norm is computed by taking
//		the maxima of the absolute values of the components.  ANN is
//		highly "coordinate-based" and does not support general distances
//		functions (e.g. those obeying just the triangle inequality).  It
//		also does not support distance functions based on
//		inner-products.
//
//		For the purpose of computing nearest neighbors, it is not
//		necessary to compute the final power (1/p).  Thus the only
//		component that is used by the program is |v(i)|^p.
//
//		ANN parameterizes the distance computation through the following
//		macros.  (Macros are used rather than procedures for
//		efficiency.) Recall that the distance between two points is
//		given by the length of the vector joining them, and the length
//		or norm of a vector v is given by formula:
//
//				|v| = ROOT(POW(v0) # POW(v1) # ... # POW(v(d-1)))
//
//		where ROOT, POW are unary functions and # is an associative and
//		commutative binary operator mapping the following types:
//
//			**	POW:	ANNcoord				--> ANNdist
//			**	#:		ANNdist x ANNdist		--> ANNdist
//			**	ROOT:	ANNdist (>0)			--> double
//
//		For early termination in distance calculation (partial distance
//		calculation) we assume that POW and # together are monotonically
//		increasing on sequences of arguments, meaning that for all
//		v0..vk and y:
//
//		POW(v0) #...# POW(vk) <= (POW(v0) #...# POW(vk)) # POW(y).
//
//	Incremental Distance Calculation:
//		The program uses an optimized method of computing distances for
//		kd-trees and bd-trees, called incremental distance calculation.
//		It is used when distances are to be updated when only a single
//		coordinate of a point has been changed.  In order to use this,
//		we assume that there is an incremental update function DIFF(x,y)
//		for #, such that if:
//
//					s = x0 # ... # xi # ... # xk
//
//		then if s' is equal to s but with xi replaced by y, that is,
//
//					s' = x0 # ... # y # ... # xk
//
//		then the length of s' can be computed by:
//
//					|s'| = |s| # DIFF(xi,y).
//
//		Thus, if # is + then DIFF(xi,y) is (yi-x).  For the L_infinity
//		norm we make use of the fact that in the program this function
//		is only invoked when y > xi, and hence DIFF(xi,y)=y.
//
//		Finally, for approximate nearest neighbor queries we assume
//		that POW and ROOT are related such that
//
//					v*ROOT(x) = ROOT(POW(v)*x)
//
//		Here are the values for the various Minkowski norms:
//
//		L_p:	p even:							p odd:
//				-------------------------		------------------------
//				POW(v)			= v^p			POW(v)			= |v|^p
//				ROOT(x)			= x^(1/p)		ROOT(x)			= x^(1/p)
//				#				= +				#				= +
//				DIFF(x,y)		= y - x			DIFF(x,y)		= y - x
//
//		L_inf:
//				POW(v)			= |v|
//				ROOT(x)			= x
//				#				= max
//				DIFF(x,y)		= y
//
//		By default the Euclidean norm is assumed.  To change the norm,
//		uncomment the appropriate set of macros below.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Use the following for the Euclidean norm
//----------------------------------------------------------------------
#define ANN_POW(v) ((v) * (v))
#define ANN_ROOT(x) sqrt(x)
#define ANN_SUM(x, y) ((x) + (y))
#define ANN_DIFF(x, y) ((y) - (x))

//----------------------------------------------------------------------
//	Use the following for the L_1 (Manhattan) norm
//----------------------------------------------------------------------
// #define ANN_POW(v)		fabs(v)
// #define ANN_ROOT(x)		(x)
// #define ANN_SUM(x,y)		((x) + (y))
// #define ANN_DIFF(x,y)	((y) - (x))

//----------------------------------------------------------------------
//	Use the following for a general L_p norm
//----------------------------------------------------------------------
// #define ANN_POW(v)		pow(fabs(v),p)
// #define ANN_ROOT(x)		pow(fabs(x),1/p)
// #define ANN_SUM(x,y)		((x) + (y))
// #define ANN_DIFF(x,y)	((y) - (x))

//----------------------------------------------------------------------
//	Use the following for the L_infinity (Max) norm
//----------------------------------------------------------------------
// #define ANN_POW(v)		fabs(v)
// #define ANN_ROOT(x)		(x)
// #define ANN_SUM(x,y)		((x) > (y) ? (x) : (y))
// #define ANN_DIFF(x,y)	(y)

//----------------------------------------------------------------------
//	Array types
//		The following array types are of basic interest.  A point is
//		just a dimensionless array of coordinates, a point array is a
//		dimensionless array of points.  A distance array is a
//		dimensionless array of distances and an index array is a
//		dimensionless array of point indices.  The latter two are used
//		when returning the results of k-nearest neighbor queries.
//----------------------------------------------------------------------

typedef ANNcoord *ANNpoint;      // a point
typedef ANNpoint *ANNpointArray; // an array of points
typedef ANNdist *ANNdistArray;   // an array of distances
typedef ANNidx *ANNidxArray;     // an array of point indices

//----------------------------------------------------------------------
//	Basic point and array utilities:
//		The following procedures are useful supplements to ANN's nearest
//		neighbor capabilities.
//
//		annDist():
//			Computes the (squared) distance between a pair of points.
//			Note that this routine is not used internally by ANN for
//			computing distance calculations.  For reasons of efficiency
//			this is done using incremental distance calculation.  Thus,
//			this routine cannot be modified as a method of changing the
//			metric.
//
//		Because points (somewhat like strings in C) are stored as
//		pointers.  Consequently, creating and destroying copies of
//		points may require storage allocation.  These procedures do
//		this.
//
//		annAllocPt() and annDeallocPt():
//				Allocate a deallocate storage for a single point, and
//				return a pointer to it.  The argument to AllocPt() is
//				used to initialize all components.
//
//		annAllocPts() and annDeallocPts():
//				Allocate and deallocate an array of points as well a
//				place to store their coordinates, and initializes the
//				points to point to their respective coordinates.  It
//				allocates point storage in a contiguous block large
//				enough to store all the points.  It performs no
//				initialization.
//
//		annCopyPt():
//				Creates a copy of a given point, allocating space for
//				the new point.  It returns a pointer to the newly
//				allocated copy.
//----------------------------------------------------------------------

DLL_API ANNdist annDist(int dim,    // dimension of space
                        ANNpoint p, // points
                        ANNpoint q);

DLL_API ANNpoint annAllocPt(int dim,         // dimension
                            ANNcoord c = 0); // coordinate value (all equal)

DLL_API ANNpointArray annAllocPts(int n,    // number of points
                                  int dim); // dimension

DLL_API void annDeallocPt(ANNpoint &p); // deallocate 1 point

DLL_API void annDeallocPts(ANNpointArray &pa); // point array

DLL_API ANNpoint annCopyPt(int dim,          // dimension
                           ANNpoint source); // point to copy

//----------------------------------------------------------------------
// Overall structure: ANN supports a number of different data structures
// for approximate and exact nearest neighbor searching.  These are:
//
//		ANNbruteForce	A simple brute-force search structure.
//		ANNkd_tree		A kd-tree tree search structure.  ANNbd_tree
//		A bd-tree tree search structure (a kd-tree with shrink
//		capabilities).
//
//		At a minimum, each of these data structures support k-nearest
//		neighbor queries.  The nearest neighbor query, annkSearch,
//		returns an integer identifier and the distance to the nearest
//		neighbor(s) and annRangeSearch returns the nearest points that
//		lie within a given query ball.
//
//		Each structure is built by invoking the appropriate constructor
//		and passing it (at a minimum) the array of points, the total
//		number of points and the dimension of the space.  Each structure
//		is also assumed to support a destructor and member functions
//		that return basic information about the point set.
//
//		Note that the array of points is not copied by the data
//		structure (for reasons of space efficiency), and it is assumed
//		to be constant throughout the lifetime of the search structure.
//
//		The search algorithm, annkSearch, is given the query point (q),
//		and the desired number of nearest neighbors to report (k), and
//		the error bound (eps) (whose default value is 0, implying exact
//		nearest neighbors).  It returns two arrays which are assumed to
//		contain at least k elements: one (nn_idx) contains the indices
//		(within the point array) of the nearest neighbors and the other
//		(dd) contains the squared distances to these nearest neighbors.
//
//		The search algorithm, annkFRSearch, is a fixed-radius kNN
//		search.  In addition to a query point, it is given a (squared)
//		radius bound.  (This is done for consistency, because the search
//		returns distances as squared quantities.) It does two things.
//		First, it computes the k nearest neighbors within the radius
//		bound, and second, it returns the total number of points lying
//		within the radius bound. It is permitted to set k = 0, in which
//		case it effectively answers a range counting query.  If the
//		error bound epsilon is positive, then the search is approximate
//		in the sense that it is free to ignore any point that lies
//		outside a ball of radius r/(1+epsilon), where r is the given
//		(unsquared) radius bound.
//
//		The generic object from which all the search structures are
//		dervied is given below.  It is a virtual object, and is useless
//		by itself.
//----------------------------------------------------------------------

class DLL_API ANNpointSet {
public:
    virtual ~ANNpointSet() {} // virtual distructor

    virtual void annkSearch( // approx k near neighbor search
        ANNpoint q,          // query point
        int k,               // number of near neighbors to return
        ANNidxArray nn_idx,  // nearest neighbor array (modified)
        ANNdistArray dd,     // dist to near neighbors (modified)
        double eps = 0.0     // error bound
        ) = 0;               // pure virtual (defined elsewhere)

    virtual int annkFRSearch(      // approx fixed-radius kNN search
        ANNpoint q,                // query point
        ANNdist sqRad,             // squared radius
        int k              = 0,    // number of near neighbors to return
        ANNidxArray nn_idx = NULL, // nearest neighbor array (modified)
        ANNdistArray dd    = NULL, // dist to near neighbors (modified)
        double eps         = 0.0   // error bound
        ) = 0;                     // pure virtual (defined elsewhere)

    virtual int theDim()  = 0; // return dimension of space
    virtual int nPoints() = 0; // return number of points
                               // return pointer to points
    virtual ANNpointArray thePoints() = 0;
};

//----------------------------------------------------------------------
//	Brute-force nearest neighbor search:
//		The brute-force search structure is very simple but inefficient.
//		It has been provided primarily for the sake of comparison with
//		and validation of the more complex search structures.
//
//		Query processing is the same as described above, but the value
//		of epsilon is ignored, since all distance calculations are
//		performed exactly.
//
//		WARNING: This data structure is very slow, and should not be
//		used unless the number of points is very small.
//
//		Internal information:
//		---------------------
//		This data structure bascially consists of the array of points
//		(each a pointer to an array of coordinates).  The search is
//		performed by a simple linear scan of all the points.
//----------------------------------------------------------------------

class DLL_API ANNbruteForce : public ANNpointSet {
    int dim;           // dimension
    int n_pts;         // number of points
    ANNpointArray pts; // point array
public:
    ANNbruteForce(        // constructor from point array
        ANNpointArray pa, // point array
        int n,            // number of points
        int dd);          // dimension

    ~ANNbruteForce(); // destructor

    void annkSearch(        // approx k near neighbor search
        ANNpoint q,         // query point
        int k,              // number of near neighbors to return
        ANNidxArray nn_idx, // nearest neighbor array (modified)
        ANNdistArray dd,    // dist to near neighbors (modified)
        double eps = 0.0);  // error bound

    int annkFRSearch(              // approx fixed-radius kNN search
        ANNpoint q,                // query point
        ANNdist sqRad,             // squared radius
        int k              = 0,    // number of near neighbors to return
        ANNidxArray nn_idx = NULL, // nearest neighbor array (modified)
        ANNdistArray dd    = NULL, // dist to near neighbors (modified)
        double eps         = 0.0);         // error bound

    int theDim() // return dimension of space
    {
        return dim;
    }

    int nPoints() // return number of points
    {
        return n_pts;
    }

    ANNpointArray thePoints() // return pointer to points
    {
        return pts;
    }
};

//----------------------------------------------------------------------
// kd- and bd-tree splitting and shrinking rules
//		kd-trees supports a collection of different splitting rules.
//		In addition to the standard kd-tree splitting rule proposed
//		by Friedman, Bentley, and Finkel, we have introduced a
//		number of other splitting rules, which seem to perform
//		as well or better (for the distributions we have tested).
//
//		The splitting methods given below allow the user to tailor
//		the data structure to the particular data set.  They are
//		are described in greater details in the kd_split.cc source
//		file.  The method ANN_KD_SUGGEST is the method chosen (rather
//		subjectively) by the implementors as the one giving the
//		fastest performance, and is the default splitting method.
//
//		As with splitting rules, there are a number of different
//		shrinking rules.  The shrinking rule ANN_BD_NONE does no
//		shrinking (and hence produces a kd-tree tree).  The rule
//		ANN_BD_SUGGEST uses the implementors favorite rule.
//----------------------------------------------------------------------

enum ANNsplitRule {
    ANN_KD_STD      = 0, // the optimized kd-splitting rule
    ANN_KD_MIDPT    = 1, // midpoint split
    ANN_KD_FAIR     = 2, // fair split
    ANN_KD_SL_MIDPT = 3, // sliding midpoint splitting method
    ANN_KD_SL_FAIR  = 4, // sliding fair split method
    ANN_KD_SUGGEST  = 5
};                               // the authors' suggestion for best
const int ANN_N_SPLIT_RULES = 6; // number of split rules

enum ANNshrinkRule {
    ANN_BD_NONE     = 0, // no shrinking at all (just kd-tree)
    ANN_BD_SIMPLE   = 1, // simple splitting
    ANN_BD_CENTROID = 2, // centroid splitting
    ANN_BD_SUGGEST  = 3
};                                // the authors' suggested choice
const int ANN_N_SHRINK_RULES = 4; // number of shrink rules

//----------------------------------------------------------------------
//	kd-tree:
//		The main search data structure supported by ANN is a kd-tree.
//		The main constructor is given a set of points and a choice of
//		splitting method to use in building the tree.
//
//		Construction:
//		-------------
//		The constructor is given the point array, number of points,
//		dimension, bucket size (default = 1), and the splitting rule
//		(default = ANN_KD_SUGGEST).  The point array is not copied, and
//		is assumed to be kept constant throughout the lifetime of the
//		search structure.  There is also a "load" constructor that
//		builds a tree from a file description that was created by the
//		Dump operation.
//
//		Search:
//		-------
//		There are two search methods:
//
//			Standard search (annkSearch()):
//				Searches nodes in tree-traversal order, always visiting
//				the closer child first.
//			Priority search (annkPriSearch()):
//				Searches nodes in order of increasing distance of the
//				associated cell from the query point.  For many
//				distributions the standard search seems to work just
//				fine, but priority search is safer for worst-case
//				performance.
//
//		Printing:
//		---------
//		There are two methods provided for printing the tree.  Print()
//		is used to produce a "human-readable" display of the tree, with
//		indenation, which is handy for debugging.  Dump() produces a
//		format that is suitable reading by another program.  There is a
//		"load" constructor, which constructs a tree which is assumed to
//		have been saved by the Dump() procedure.
//
//		Performance and Structure Statistics:
//		-------------------------------------
//		The procedure getStats() collects statistics information on the
//		tree (its size, height, etc.)  See ANNperf.h for information on
//		the stats structure it returns.
//
//		Internal information:
//		---------------------
//		The data structure consists of three major chunks of storage.
//		The first (implicit) storage are the points themselves (pts),
//		which have been provided by the users as an argument to the
//		constructor, or are allocated dynamically if the tree is built
//		using the load constructor).  These should not be changed during
//		the lifetime of the search structure.  It is the user's
//		responsibility to delete these after the tree is destroyed.
//
//		The second is the tree itself (which is dynamically allocated in
//		the constructor) and is given as a pointer to its root node
//		(root).  These nodes are automatically deallocated when the tree
//		is deleted.  See the file src/kd_tree.h for further information
//		on the structure of the tree nodes.
//
//		Each leaf of the tree does not contain a pointer directly to a
//		point, but rather contains a pointer to a "bucket", which is an
//		array consisting of point indices.  The third major chunk of
//		storage is an array (pidx), which is a large array in which all
//		these bucket subarrays reside.  (The reason for storing them
//		separately is the buckets are typically small, but of varying
//		sizes.  This was done to avoid fragmentation.)  This array is
//		also deallocated when the tree is deleted.
//
//		In addition to this, the tree consists of a number of other
//		pieces of information which are used in searching and for
//		subsequent tree operations.  These consist of the following:
//
//		dim						Dimension of space
//		n_pts					Number of points currently in the tree
//		n_max					Maximum number of points that are allowed
//								in the tree
//		bkt_size				Maximum bucket size (no. of points per leaf)
//		bnd_box_lo				Bounding box low point
//		bnd_box_hi				Bounding box high point
//		splitRule				Splitting method used
//
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Some types and objects used by kd-tree functions
// See src/kd_tree.h and src/kd_tree.cpp for definitions
//----------------------------------------------------------------------
class ANNkdStats;              // stats on kd-tree
class ANNkd_node;              // generic node in a kd-tree
typedef ANNkd_node *ANNkd_ptr; // pointer to a kd-tree node

class DLL_API ANNkd_tree : public ANNpointSet {
protected:
    int dim;             // dimension of space
    int n_pts;           // number of points in tree
    int bkt_size;        // bucket size
    ANNpointArray pts;   // the points
    ANNidxArray pidx;    // point indices (to pts array)
    ANNkd_ptr root;      // root of kd-tree
    ANNpoint bnd_box_lo; // bounding box low point
    ANNpoint bnd_box_hi; // bounding box high point

    void SkeletonTree(           // construct skeleton tree
        int n,                   // number of points
        int dd,                  // dimension
        int bs,                  // bucket size
        ANNpointArray pa = NULL, // point array (optional)
        ANNidxArray pi   = NULL);  // point indices (optional)

public:
    ANNkd_tree(      // build skeleton tree
        int n  = 0,  // number of points
        int dd = 0,  // dimension
        int bs = 1); // bucket size

    ANNkd_tree(                               // build from point array
        ANNpointArray pa,                     // point array
        int n,                                // number of points
        int dd,                               // dimension
        int bs             = 1,               // bucket size
        ANNsplitRule split = ANN_KD_SUGGEST); // splitting method

    ANNkd_tree(            // build from dump file
        std::istream &in); // input stream for dump file

    ~ANNkd_tree(); // tree destructor

    void annkSearch(        // approx k near neighbor search
        ANNpoint q,         // query point
        int k,              // number of near neighbors to return
        ANNidxArray nn_idx, // nearest neighbor array (modified)
        ANNdistArray dd,    // dist to near neighbors (modified)
        double eps = 0.0);  // error bound

    void annkPriSearch(     // priority k near neighbor search
        ANNpoint q,         // query point
        int k,              // number of near neighbors to return
        ANNidxArray nn_idx, // nearest neighbor array (modified)
        ANNdistArray dd,    // dist to near neighbors (modified)
        double eps = 0.0);  // error bound

    int annkFRSearch(              // approx fixed-radius kNN search
        ANNpoint q,                // the query point
        ANNdist sqRad,             // squared radius of query ball
        int k,                     // number of neighbors to return
        ANNidxArray nn_idx = NULL, // nearest neighbor array (modified)
        ANNdistArray dd    = NULL, // dist to near neighbors (modified)
        double eps         = 0.0);         // error bound

    int theDim() // return dimension of space
    {
        return dim;
    }

    int nPoints() // return number of points
    {
        return n_pts;
    }

    ANNpointArray thePoints() // return pointer to points
    {
        return pts;
    }

    virtual void Print(     // print the tree (for debugging)
        ANNbool with_pts,   // print points as well?
        std::ostream &out); // output stream

    virtual void Dump(      // dump entire tree
        ANNbool with_pts,   // print points as well?
        std::ostream &out); // output stream

    virtual void getStats( // compute tree statistics
        ANNkdStats &st);   // the statistics (modified)
};

//----------------------------------------------------------------------
//	Box decomposition tree (bd-tree)
//		The bd-tree is inherited from a kd-tree.  The main difference
//		in the bd-tree and the kd-tree is a new type of internal node
//		called a shrinking node (in the kd-tree there is only one type
//		of internal node, a splitting node).  The shrinking node
//		makes it possible to generate balanced trees in which the
//		cells have bounded aspect ratio, by allowing the decomposition
//		to zoom in on regions of dense point concentration.  Although
//		this is a nice idea in theory, few point distributions are so
//		densely clustered that this is really needed.
//----------------------------------------------------------------------

class DLL_API ANNbd_tree : public ANNkd_tree {
public:
    ANNbd_tree(                    // build skeleton tree
        int n,                     // number of points
        int dd,                    // dimension
        int bs = 1)                // bucket size
        : ANNkd_tree(n, dd, bs) {} // build base kd-tree

    ANNbd_tree(                                 // build from point array
        ANNpointArray pa,                       // point array
        int n,                                  // number of points
        int dd,                                 // dimension
        int bs               = 1,               // bucket size
        ANNsplitRule split   = ANN_KD_SUGGEST,  // splitting rule
        ANNshrinkRule shrink = ANN_BD_SUGGEST); // shrinking rule

    ANNbd_tree(            // build from dump file
        std::istream &in); // input stream for dump file
};

//----------------------------------------------------------------------
//	Other functions
//	annMaxPtsVisit		Sets a limit on the maximum number of points
//						to visit in the search.
//  annClose			Can be called when all use of ANN is finished.
//						It clears up a minor memory leak.
//----------------------------------------------------------------------

DLL_API void annMaxPtsVisit( // max. pts to visit in search
    int maxPts);             // the limit

DLL_API void annClose(); // called to end use of ANN

#endif

//----------------------------------------------------------------------
//	File:			ANNperf.h
//	Programmer:		Sunil Arya and David Mount
//	Last modified:	03/04/98 (Release 0.1)
//	Description:	Include file for ANN performance stats
//
//	Some of the code for statistics gathering has been adapted
//	from the SmplStat.h package in the g++ library.
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
//      History:
//      Revision 0.1  03/04/98
//          Initial release
//      Revision 1.0  04/01/05
//          Added ANN_ prefix to avoid name conflicts.
//----------------------------------------------------------------------

#ifndef ANNperf_H
#define ANNperf_H

//----------------------------------------------------------------------
//	basic includes
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// kd-tree stats object
//	This object is used for collecting information about a kd-tree
//	or bd-tree.
//----------------------------------------------------------------------

class ANNkdStats { // stats on kd-tree
public:
    int dim;      // dimension of space
    int n_pts;    // no. of points
    int bkt_size; // bucket size
    int n_lf;     // no. of leaves (including trivial)
    int n_tl;     // no. of trivial leaves (no points)
    int n_spl;    // no. of splitting nodes
    int n_shr;    // no. of shrinking nodes (for bd-trees)
    int depth;    // depth of tree
    float sum_ar; // sum of leaf aspect ratios
    float avg_ar; // average leaf aspect ratio
    //
    // reset stats
    void reset(int d = 0, int n = 0, int bs = 0) {
        dim      = d;
        n_pts    = n;
        bkt_size = bs;
        n_lf = n_tl = n_spl = n_shr = depth = 0;
        sum_ar = avg_ar = 0.0;
    }

    ANNkdStats() // basic constructor
    {
        reset();
    }

    void merge(const ANNkdStats &st); // merge stats from child
};

//----------------------------------------------------------------------
//  ANNsampStat
//	A sample stat collects numeric (double) samples and returns some
//	simple statistics.  Its main functions are:
//
//		reset()		Reset to no samples.
//		+= x		Include sample x.
//		samples()	Return number of samples.
//		mean()		Return mean of samples.
//		stdDev()	Return standard deviation
//		min()		Return minimum of samples.
//		max()		Return maximum of samples.
//----------------------------------------------------------------------
class DLL_API ANNsampStat {
    int n;                 // number of samples
    double sum;            // sum
    double sum2;           // sum of squares
    double minVal, maxVal; // min and max
public:
    void reset() // reset everything
    {
        n   = 0;
        sum = sum2 = 0;
        minVal     = ANN_DBL_MAX;
        maxVal     = -ANN_DBL_MAX;
    }

    ANNsampStat() { reset(); } // constructor

    void operator+=(double x) // add sample
    {
        n++;
        sum += x;
        sum2 += x * x;
        if (x < minVal)
            minVal = x;
        if (x > maxVal)
            maxVal = x;
    }

    int samples() { return n; } // number of samples

    double mean() { return sum / n; } // mean

    // standard deviation
    double stdDev() { return sqrt((sum2 - (sum * sum) / n) / (n - 1)); }

    double min() { return minVal; } // minimum
    double max() { return maxVal; } // maximum
};

//----------------------------------------------------------------------
//		Operation count updates
//----------------------------------------------------------------------

#ifdef ANN_PERF
#define ANN_FLOP(n)                                                                                                    \
    { ann_Nfloat_ops += (n); }
#define ANN_LEAF(n)                                                                                                    \
    { ann_Nvisit_lfs += (n); }
#define ANN_SPL(n)                                                                                                     \
    { ann_Nvisit_spl += (n); }
#define ANN_SHR(n)                                                                                                     \
    { ann_Nvisit_shr += (n); }
#define ANN_PTS(n)                                                                                                     \
    { ann_Nvisit_pts += (n); }
#define ANN_COORD(n)                                                                                                   \
    { ann_Ncoord_hts += (n); }
#else
#define ANN_FLOP(n)
#define ANN_LEAF(n)
#define ANN_SPL(n)
#define ANN_SHR(n)
#define ANN_PTS(n)
#define ANN_COORD(n)
#endif

//----------------------------------------------------------------------
//	Performance statistics
//	The following data and routines are used for computing performance
//	statistics for nearest neighbor searching.  Because these routines
//	can slow the code down, they can be activated and deactiviated by
//	defining the ANN_PERF variable, by compiling with the option:
//	-DANN_PERF
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Global counters for performance measurement
//
//	visit_lfs	The number of leaf nodes visited in the
//				tree.
//
//	visit_spl	The number of splitting nodes visited in the
//				tree.
//
//	visit_shr	The number of shrinking nodes visited in the
//				tree.
//
//	visit_pts	The number of points visited in all the
//				leaf nodes visited. Equivalently, this
//				is the number of points for which distance
//				calculations are performed.
//
//	coord_hts	The number of times a coordinate of a
//				data point is accessed. This is generally
//				less than visit_pts*d if partial distance
//				calculation is used.  This count is low
//				in the sense that if a coordinate is hit
//				many times in the same routine we may
//				count it only once.
//
//	float_ops	The number of floating point operations.
//				This includes all operations in the heap
//				as well as distance calculations to boxes.
//
//	average_err	The average error of each query (the
//				error of the reported point to the true
//				nearest neighbor).  For k nearest neighbors
//				the error is computed k times.
//
//	rank_err	The rank error of each query (the difference
//				in the rank of the reported point and its
//				true rank).
//
//	data_pts	The number of data points.  This is not
//				a counter, but used in stats computation.
//----------------------------------------------------------------------

extern int ann_Ndata_pts;         // number of data points
extern int ann_Nvisit_lfs;        // number of leaf nodes visited
extern int ann_Nvisit_spl;        // number of splitting nodes visited
extern int ann_Nvisit_shr;        // number of shrinking nodes visited
extern int ann_Nvisit_pts;        // visited points for one query
extern int ann_Ncoord_hts;        // coordinate hits for one query
extern int ann_Nfloat_ops;        // floating ops for one query
extern ANNsampStat ann_visit_lfs; // stats on leaf nodes visits
extern ANNsampStat ann_visit_spl; // stats on splitting nodes visits
extern ANNsampStat ann_visit_shr; // stats on shrinking nodes visits
extern ANNsampStat ann_visit_nds; // stats on total nodes visits
extern ANNsampStat ann_visit_pts; // stats on points visited
extern ANNsampStat ann_coord_hts; // stats on coordinate hits
extern ANNsampStat ann_float_ops; // stats on floating ops
//----------------------------------------------------------------------
//  The following need to be part of the public interface, because
//  they are accessed outside the DLL in ann_test.cpp.
//----------------------------------------------------------------------
DLL_API extern ANNsampStat ann_average_err; // average error
DLL_API extern ANNsampStat ann_rank_err;    // rank error

//----------------------------------------------------------------------
//	Declaration of externally accessible routines for statistics
//----------------------------------------------------------------------

DLL_API void annResetStats(int data_size); // reset stats for a set of queries

DLL_API void annResetCounts(); // reset counts for one queries

DLL_API void annUpdateStats(); // update stats with current counts

DLL_API void annPrintStats(ANNbool validate); // print statistics for a run

#endif

//----------------------------------------------------------------------
// File:			ANNx.h
// Programmer: 		Sunil Arya and David Mount
// Description:		Internal include file for ANN
// Last modified:	01/27/10 (Version 1.1.2)
//
//	These declarations are of use in manipulating some of
//	the internal data objects appearing in ANN, but are not
//	needed for applications just using the nearest neighbor
//	search.
//
//	Typical users of ANN should not need to access this file.
//----------------------------------------------------------------------
// Copyright (c) 1997-2010 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
//	History:
//	Revision 0.1  03/04/98
//	    Initial release
//	Revision 1.0  04/01/05
//	    Changed LO, HI, IN, OUT to ANN_LO, ANN_HI, etc.
//	Revision 1.1.2  01/27/10
//		Fixed minor compilation bugs for new versions of gcc
//----------------------------------------------------------------------

#ifndef ANNx_H
#define ANNx_H

#include <iomanip> // I/O manipulators

//----------------------------------------------------------------------
//	Global constants and types
//----------------------------------------------------------------------
enum { ANN_LO = 0, ANN_HI = 1 }; // splitting indices
enum {
    ANN_IN  = 0,
    ANN_OUT = 1
}; // shrinking indices
   // what to do in case of error
enum ANNerr { ANNwarn = 0, ANNabort = 1 };

//----------------------------------------------------------------------
//	Maximum number of points to visit
//	We have an option for terminating the search early if the
//	number of points visited exceeds some threshold.  If the
//	threshold is 0 (its default)  this means there is no limit
//	and the algorithm applies its normal termination condition.
//----------------------------------------------------------------------

extern int ANNmaxPtsVisited; // maximum number of pts visited
extern int ANNptsVisited;    // number of pts visited in search

//----------------------------------------------------------------------
//	Global function declarations
//----------------------------------------------------------------------

void annError(       // ANN error routine
    const char *msg, // error message
    ANNerr level);   // level of error

void annPrintPt(        // print a point
    ANNpoint pt,        // the point
    int dim,            // the dimension
    std::ostream &out); // output stream

//----------------------------------------------------------------------
//	Orthogonal (axis aligned) rectangle
//	Orthogonal rectangles are represented by two points, one
//	for the lower left corner (min coordinates) and the other
//	for the upper right corner (max coordinates).
//
//	The constructor initializes from either a pair of coordinates,
//	pair of points, or another rectangle.  Note that all constructors
//	allocate new point storage. The destructor deallocates this
//	storage.
//
//	BEWARE: Orthogonal rectangles should be passed ONLY BY REFERENCE.
//	(C++'s default copy constructor will not allocate new point
//	storage, then on return the destructor free's storage, and then
//	you get into big trouble in the calling procedure.)
//----------------------------------------------------------------------

class ANNorthRect {
public:
    ANNpoint lo;        // rectangle lower bounds
    ANNpoint hi;        // rectangle upper bounds
                        //
    ANNorthRect(        // basic constructor
        int dd,         // dimension of space
        ANNcoord l = 0, // default is empty
        ANNcoord h = 0) {
        lo = annAllocPt(dd, l);
        hi = annAllocPt(dd, h);
    }

    ANNorthRect(              // (almost a) copy constructor
        int dd,               // dimension
        const ANNorthRect &r) // rectangle to copy
    {
        lo = annCopyPt(dd, r.lo);
        hi = annCopyPt(dd, r.hi);
    }

    ANNorthRect(    // construct from points
        int dd,     // dimension
        ANNpoint l, // low point
        ANNpoint h) // hight point
    {
        lo = annCopyPt(dd, l);
        hi = annCopyPt(dd, h);
    }

    ~ANNorthRect() // destructor
    {
        annDeallocPt(lo);
        annDeallocPt(hi);
    }

    ANNbool inside(int dim, ANNpoint p); // is point p inside rectangle?
};

void annAssignRect(             // assign one rect to another
    int dim,                    // dimension (both must be same)
    ANNorthRect &dest,          // destination (modified)
    const ANNorthRect &source); // source

//----------------------------------------------------------------------
//	Orthogonal (axis aligned) halfspace
//	An orthogonal halfspace is represented by an integer cutting
//	dimension cd, coordinate cutting value, cv, and side, sd, which is
//	either +1 or -1. Our convention is that point q lies in the (closed)
//	halfspace if (q[cd] - cv)*sd >= 0.
//----------------------------------------------------------------------

class ANNorthHalfSpace {
public:
    int cd;            // cutting dimension
    ANNcoord cv;       // cutting value
    int sd;            // which side
                       //
    ANNorthHalfSpace() // default constructor
    {
        cd = 0;
        cv = 0;
        sd = 0;
    }

    ANNorthHalfSpace( // basic constructor
        int cdd,      // dimension of space
        ANNcoord cvv, // cutting value
        int sdd)      // side
    {
        cd = cdd;
        cv = cvv;
        sd = sdd;
    }

    ANNbool in(ANNpoint q) const // is q inside halfspace?
    {
        return (ANNbool) ((q[cd] - cv) * sd >= 0);
    }

    ANNbool out(ANNpoint q) const // is q outside halfspace?
    {
        return (ANNbool) ((q[cd] - cv) * sd < 0);
    }

    ANNdist dist(ANNpoint q) const // (squared) distance from q
    {
        return (ANNdist) ANN_POW(q[cd] - cv);
    }

    void setLowerBound(int d, ANNpoint p) // set to lower bound at p[i]
    {
        cd = d;
        cv = p[d];
        sd = +1;
    }

    void setUpperBound(int d, ANNpoint p) // set to upper bound at p[i]
    {
        cd = d;
        cv = p[d];
        sd = -1;
    }

    void project(ANNpoint &q) // project q (modified) onto halfspace
    {
        if (out(q))
            q[cd] = cv;
    }
};

// array of halfspaces
typedef ANNorthHalfSpace *ANNorthHSArray;

#endif

//----------------------------------------------------------------------
// File:			kd_tree.h
// Programmer:		Sunil Arya and David Mount
// Description:		Declarations for standard kd-tree routines
// Last modified:	05/03/05 (Version 1.1)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.1  05/03/05
//		Added fixed radius kNN search
//----------------------------------------------------------------------

#ifndef ANN_kd_tree_H
#define ANN_kd_tree_H

//----------------------------------------------------------------------
//	Generic kd-tree node
//
//		Nodes in kd-trees are of two types, splitting nodes which contain
//		splitting information (a splitting hyperplane orthogonal to one
//		of the coordinate axes) and leaf nodes which contain point
//		information (an array of points stored in a bucket).  This is
//		handled by making a generic class kd_node, which is essentially an
//		empty shell, and then deriving the leaf and splitting nodes from
//		this.
//----------------------------------------------------------------------

class ANNmin_k;

struct ANNkdQueryInfo {
    int ANNkdDim;           // dimension of space
    ANNpoint ANNkdQ;        // query point
    double ANNkdMaxErr;     // max tolerable squared error
    ANNpointArray ANNkdPts; // the points
    ANNmin_k *ANNkdPointMK; // set of k closest points
};

class ANNkd_node { // generic kd-tree node (empty shell)
public:
    virtual ~ANNkd_node() {} // virtual distroyer

    virtual void ann_search(ANNdist, ANNkdQueryInfo &qinfo) = 0; // tree search
    virtual void ann_pri_search(ANNdist)                    = 0; // priority search
    virtual void ann_FR_search(ANNdist)                     = 0; // fixed-radius search

    virtual void getStats(         // get tree statistics
        int dim,                   // dimension of space
        ANNkdStats &st,            // statistics
        ANNorthRect &bnd_box) = 0; // bounding box
                                   // print node
    virtual void print(int level, ostream &out) = 0;
    virtual void dump(ostream &out)             = 0; // dump node

    friend class ANNkd_tree; // allow kd-tree to access us
};

//----------------------------------------------------------------------
//	kd-splitting function:
//		kd_splitter is a pointer to a splitting routine for preprocessing.
//		Different splitting procedures result in different strategies
//		for building the tree.
//----------------------------------------------------------------------

typedef void (*ANNkd_splitter)( // splitting routine for kd-trees
    ANNpointArray pa,           // point array (unaltered)
    ANNidxArray pidx,           // point indices (permuted on return)
    const ANNorthRect &bnds,    // bounding rectangle for cell
    int n,                      // number of points
    int dim,                    // dimension of space
    int &cut_dim,               // cutting dimension (returned)
    ANNcoord &cut_val,          // cutting value (returned)
    int &n_lo);                 // num of points on low side (returned)

//----------------------------------------------------------------------
//	Leaf kd-tree node
//		Leaf nodes of the kd-tree store the set of points associated
//		with this bucket, stored as an array of point indices.  These
//		are indices in the array points, which resides with the
//		root of the kd-tree.  We also store the number of points
//		that reside in this bucket.
//----------------------------------------------------------------------

class ANNkd_leaf : public ANNkd_node // leaf node for kd-tree
{
    int n_pts;       // no. points in bucket
    ANNidxArray bkt; // bucket of points
public:
    ANNkd_leaf(        // constructor
        int n,         // number of points
        ANNidxArray b) // bucket
    {
        n_pts = n; // number of points in bucket
        bkt   = b; // the bucket
    }

    ~ANNkd_leaf() {} // destructor (none)

    virtual void getStats(                       // get tree statistics
        int dim,                                 // dimension of space
        ANNkdStats &st,                          // statistics
        ANNorthRect &bnd_box);                   // bounding box
    virtual void print(int level, ostream &out); // print node
    virtual void dump(ostream &out);             // dump node

    virtual void ann_search(ANNdist, ANNkdQueryInfo &); // standard search
    virtual void ann_pri_search(ANNdist);               // priority search
    virtual void ann_FR_search(ANNdist);                // fixed-radius search
};

//----------------------------------------------------------------------
//		KD_TRIVIAL is a special pointer to an empty leaf node. Since
//		some splitting rules generate many (more than 50%) trivial
//		leaves, we use this one shared node to save space.
//
//		The pointer is initialized to NULL, but whenever a kd-tree is
//		created, we allocate this node, if it has not already been
//		allocated. This node is *never* deallocated, so it produces
//		a small memory leak.
//----------------------------------------------------------------------

extern ANNkd_leaf *KD_TRIVIAL; // trivial (empty) leaf node

//----------------------------------------------------------------------
//	kd-tree splitting node.
//		Splitting nodes contain a cutting dimension and a cutting value.
//		These indicate the axis-parellel plane which subdivide the
//		box for this node. The extent of the bounding box along the
//		cutting dimension is maintained (this is used to speed up point
//		to box distance calculations) [we do not store the entire bounding
//		box since this may be wasteful of space in high dimensions].
//		We also store pointers to the 2 children.
//----------------------------------------------------------------------

class ANNkd_split : public ANNkd_node // splitting node of a kd-tree
{
    int cut_dim;         // dim orthogonal to cutting plane
    ANNcoord cut_val;    // location of cutting plane
    ANNcoord cd_bnds[2]; // lower and upper bounds of
                         // rectangle along cut_dim
    ANNkd_ptr child[2];  // left and right children
public:
    ANNkd_split(                                  // constructor
        int cd,                                   // cutting dimension
        ANNcoord cv,                              // cutting value
        ANNcoord lv, ANNcoord hv,                 // low and high values
        ANNkd_ptr lc = NULL, ANNkd_ptr hc = NULL) // children
    {
        cut_dim         = cd; // cutting dimension
        cut_val         = cv; // cutting value
        cd_bnds[ANN_LO] = lv; // lower bound for rectangle
        cd_bnds[ANN_HI] = hv; // upper bound for rectangle
        child[ANN_LO]   = lc; // left child
        child[ANN_HI]   = hc; // right child
    }

    ~ANNkd_split() // destructor
    {
        if (child[ANN_LO] != NULL && child[ANN_LO] != KD_TRIVIAL)
            delete child[ANN_LO];
        if (child[ANN_HI] != NULL && child[ANN_HI] != KD_TRIVIAL)
            delete child[ANN_HI];
    }

    virtual void getStats(                       // get tree statistics
        int dim,                                 // dimension of space
        ANNkdStats &st,                          // statistics
        ANNorthRect &bnd_box);                   // bounding box
    virtual void print(int level, ostream &out); // print node
    virtual void dump(ostream &out);             // dump node

    virtual void ann_search(ANNdist, ANNkdQueryInfo &); // standard search
    virtual void ann_pri_search(ANNdist);               // priority search
    virtual void ann_FR_search(ANNdist);                // fixed-radius search
};

//----------------------------------------------------------------------
//		External entry points
//----------------------------------------------------------------------

ANNkd_ptr rkd_tree(           // recursive construction of kd-tree
    ANNpointArray pa,         // point array (unaltered)
    ANNidxArray pidx,         // point indices to store in subtree
    int n,                    // number of points
    int dim,                  // dimension of space
    int bsp,                  // bucket space
    ANNorthRect &bnd_box,     // bounding box for current node
    ANNkd_splitter splitter); // splitting routine

#endif

//----------------------------------------------------------------------
// File:			bd_tree.h
// Programmer:		David Mount
// Description:		Declarations for standard bd-tree routines
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.0  04/01/05
//		Changed IN, OUT to ANN_IN, ANN_OUT
//----------------------------------------------------------------------

#ifndef ANN_bd_tree_H
#define ANN_bd_tree_H

//----------------------------------------------------------------------
//	bd-tree shrinking node.
//		The main addition in the bd-tree is the shrinking node, which
//		is declared here.
//
//		Shrinking nodes are defined by list of orthogonal halfspaces.
//		These halfspaces define a (possibly unbounded) orthogonal
//		rectangle.  There are two children, in and out.  Points that
//		lie within this rectangle are stored in the in-child, and the
//		other points are stored in the out-child.
//
//		We use a list of orthogonal halfspaces rather than an
//		orthogonal rectangle object because typically the number of
//		sides of the shrinking box will be much smaller than the
//		worst case bound of 2*dim.
//
//		BEWARE: Note that constructor just copies the pointer to the
//		bounding array, but the destructor deallocates it.  This is
//		rather poor practice, but happens to be convenient.  The list
//		is allocated in the bd-tree building procedure rbd_tree() just
//		prior to construction, and is used for no other purposes.
//
//		WARNING: In the near neighbor searching code it is assumed that
//		the list of bounding halfspaces is irredundant, meaning that there
//		are no two distinct halfspaces in the list with the same outward
//		pointing normals.
//----------------------------------------------------------------------

class ANNbd_shrink : public ANNkd_node // splitting node of a kd-tree
{
    int n_bnds;          // number of bounding halfspaces
    ANNorthHSArray bnds; // list of bounding halfspaces
    ANNkd_ptr child[2];  // in and out children
public:
    ANNbd_shrink(                                 // constructor
        int nb,                                   // number of bounding halfspaces
        ANNorthHSArray bds,                       // list of bounding halfspaces
        ANNkd_ptr ic = NULL, ANNkd_ptr oc = NULL) // children
    {
        n_bnds         = nb;  // cutting dimension
        bnds           = bds; // assign bounds
        child[ANN_IN]  = ic;  // set children
        child[ANN_OUT] = oc;
    }

    ~ANNbd_shrink() // destructor
    {
        if (child[ANN_IN] != NULL && child[ANN_IN] != KD_TRIVIAL)
            delete child[ANN_IN];
        if (child[ANN_OUT] != NULL && child[ANN_OUT] != KD_TRIVIAL)
            delete child[ANN_OUT];
        if (bnds != NULL)
            delete[] bnds; // delete bounds
    }

    virtual void getStats(                       // get tree statistics
        int dim,                                 // dimension of space
        ANNkdStats &st,                          // statistics
        ANNorthRect &bnd_box);                   // bounding box
    virtual void print(int level, ostream &out); // print node
    virtual void dump(ostream &out);             // dump node

    virtual void ann_search(ANNdist, ANNkdQueryInfo &); // standard search
    virtual void ann_pri_search(ANNdist);               // priority search
    virtual void ann_FR_search(ANNdist);                // fixed-radius search
};

#endif
//----------------------------------------------------------------------
// File:			kd_split.h
// Programmer:		Sunil Arya and David Mount
// Description:		Methods for splitting kd-trees
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

#ifndef ANN_KD_SPLIT_H
#define ANN_KD_SPLIT_H

//----------------------------------------------------------------------
//	External entry points
//		These are all splitting procedures for kd-trees.
//----------------------------------------------------------------------

void kd_split(               // standard (optimized) kd-splitter
    ANNpointArray pa,        // point array (unaltered)
    ANNidxArray pidx,        // point indices (permuted on return)
    const ANNorthRect &bnds, // bounding rectangle for cell
    int n,                   // number of points
    int dim,                 // dimension of space
    int &cut_dim,            // cutting dimension (returned)
    ANNcoord &cut_val,       // cutting value (returned)
    int &n_lo);              // num of points on low side (returned)

void midpt_split(            // midpoint kd-splitter
    ANNpointArray pa,        // point array (unaltered)
    ANNidxArray pidx,        // point indices (permuted on return)
    const ANNorthRect &bnds, // bounding rectangle for cell
    int n,                   // number of points
    int dim,                 // dimension of space
    int &cut_dim,            // cutting dimension (returned)
    ANNcoord &cut_val,       // cutting value (returned)
    int &n_lo);              // num of points on low side (returned)

void sl_midpt_split(         // sliding midpoint kd-splitter
    ANNpointArray pa,        // point array (unaltered)
    ANNidxArray pidx,        // point indices (permuted on return)
    const ANNorthRect &bnds, // bounding rectangle for cell
    int n,                   // number of points
    int dim,                 // dimension of space
    int &cut_dim,            // cutting dimension (returned)
    ANNcoord &cut_val,       // cutting value (returned)
    int &n_lo);              // num of points on low side (returned)

void fair_split(             // fair-split kd-splitter
    ANNpointArray pa,        // point array (unaltered)
    ANNidxArray pidx,        // point indices (permuted on return)
    const ANNorthRect &bnds, // bounding rectangle for cell
    int n,                   // number of points
    int dim,                 // dimension of space
    int &cut_dim,            // cutting dimension (returned)
    ANNcoord &cut_val,       // cutting value (returned)
    int &n_lo);              // num of points on low side (returned)

void sl_fair_split(          // sliding fair-split kd-splitter
    ANNpointArray pa,        // point array (unaltered)
    ANNidxArray pidx,        // point indices (permuted on return)
    const ANNorthRect &bnds, // bounding rectangle for cell
    int n,                   // number of points
    int dim,                 // dimension of space
    int &cut_dim,            // cutting dimension (returned)
    ANNcoord &cut_val,       // cutting value (returned)
    int &n_lo);              // num of points on low side (returned)

#endif

//----------------------------------------------------------------------
// File:			kd_util.h
// Programmer:		Sunil Arya and David Mount
// Description:		Common utilities for kd- trees
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

#ifndef ANN_kd_util_H
#define ANN_kd_util_H

//----------------------------------------------------------------------
//	externally accessible functions
//----------------------------------------------------------------------

double annAspectRatio(           // compute aspect ratio of box
    int dim,                     // dimension
    const ANNorthRect &bnd_box); // bounding cube

void annEnclRect(       // compute smallest enclosing rectangle
    ANNpointArray pa,   // point array
    ANNidxArray pidx,   // point indices
    int n,              // number of points
    int dim,            // dimension
    ANNorthRect &bnds); // bounding cube (returned)

void annEnclCube(       // compute smallest enclosing cube
    ANNpointArray pa,   // point array
    ANNidxArray pidx,   // point indices
    int n,              // number of points
    int dim,            // dimension
    ANNorthRect &bnds); // bounding cube (returned)

ANNdist annBoxDistance( // compute distance from point to box
    const ANNpoint q,   // the point
    const ANNpoint lo,  // low point of box
    const ANNpoint hi,  // high point of box
    int dim);           // dimension of space

ANNcoord annSpread(   // compute point spread along dimension
    ANNpointArray pa, // point array
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d);           // dimension to check

void annMinMax(       // compute min and max coordinates along dim
    ANNpointArray pa, // point array
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d,            // dimension to check
    ANNcoord &min,    // minimum value (returned)
    ANNcoord &max);   // maximum value (returned)

int annMaxSpread(     // compute dimension of max spread
    ANNpointArray pa, // point array
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int dim);         // dimension of space

void annMedianSplit(  // split points along median value
    ANNpointArray pa, // points to split
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d,            // dimension along which to split
    ANNcoord &cv,     // cutting value
    int n_lo);        // split into n_lo and n-n_lo

void annPlaneSplit(   // split points by a plane
    ANNpointArray pa, // points to split
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d,            // dimension along which to split
    ANNcoord cv,      // cutting value
    int &br1,         // first break (values < cv)
    int &br2);        // second break (values == cv)

void annBoxSplit(     // split points by a box
    ANNpointArray pa, // points to split
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int dim,          // dimension of space
    ANNorthRect &box, // the box
    int &n_in);       // number of points inside (returned)

int annSplitBalance(  // determine balance factor of a split
    ANNpointArray pa, // points to split
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d,            // dimension along which to split
    ANNcoord cv);     // cutting value

void annBox2Bnds(                 // convert inner box to bounds
    const ANNorthRect &inner_box, // inner box
    const ANNorthRect &bnd_box,   // enclosing box
    int dim,                      // dimension of space
    int &n_bnds,                  // number of bounds (returned)
    ANNorthHSArray &bnds);        // bounds array (returned)

void annBnds2Box(               // convert bounds to inner box
    const ANNorthRect &bnd_box, // enclosing box
    int dim,                    // dimension of space
    int n_bnds,                 // number of bounds
    ANNorthHSArray bnds,        // bounds array
    ANNorthRect &inner_box);    // inner box (returned)

#endif

//----------------------------------------------------------------------
// File:			pr_queue_k.h
// Programmer:		Sunil Arya and David Mount
// Description:		Include file for priority queue with k items.
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

#ifndef PR_QUEUE_K_H
#define PR_QUEUE_K_H

//----------------------------------------------------------------------
//	Basic types
//----------------------------------------------------------------------
typedef ANNdist PQKkey; // key field is distance
typedef int PQKinfo;    // info field is int

//----------------------------------------------------------------------
//	Constants
//		The NULL key value is used to initialize the priority queue, and
//		so it should be larger than any valid distance, so that it will
//		be replaced as legal distance values are inserted.  The NULL
//		info value must be a nonvalid array index, we use ANN_NULL_IDX,
//		which is guaranteed to be negative.
//----------------------------------------------------------------------

const PQKkey PQ_NULL_KEY   = ANN_DIST_INF; // nonexistent key value
const PQKinfo PQ_NULL_INFO = ANN_NULL_IDX; // nonexistent info value

//----------------------------------------------------------------------
//	ANNmin_k
//		An ANNmin_k structure is one which maintains the smallest
//		k values (of type PQKkey) and associated information (of type
//		PQKinfo).  The special info and key values PQ_NULL_INFO and
//		PQ_NULL_KEY means that thise entry is empty.
//
//		It is currently implemented using an array with k items.
//		Items are stored in increasing sorted order, and insertions
//		are made through standard insertion sort.  (This is quite
//		inefficient, but current applications call for small values
//		of k and relatively few insertions.)
//
//		Note that the list contains k+1 entries, but the last entry
//		is used as a simple placeholder and is otherwise ignored.
//----------------------------------------------------------------------

class ANNmin_k {
    struct mk_node {  // node in min_k structure
        PQKkey key;   // key value
        PQKinfo info; // info field (user defined)
    };

    int k;       // max number of keys to store
    int n;       // number of keys currently active
    mk_node *mk; // the list itself

public:
    ANNmin_k(int max) // constructor (given max size)
    {
        n  = 0;                    // initially no items
        k  = max;                  // maximum number of items
        mk = new mk_node[max + 1]; // sorted array of keys
    }

    ~ANNmin_k() // destructor
    {
        delete[] mk;
    }

    PQKkey ANNmin_key() // return minimum key
    {
        return (n > 0 ? mk[0].key : PQ_NULL_KEY);
    }

    PQKkey max_key() // return maximum key
    {
        return (n == k ? mk[k - 1].key : PQ_NULL_KEY);
    }

    PQKkey ith_smallest_key(int i) // ith smallest key (i in [0..n-1])
    {
        return (i < n ? mk[i].key : PQ_NULL_KEY);
    }

    PQKinfo ith_smallest_info(int i) // info for ith smallest (i in [0..n-1])
    {
        return (i < n ? mk[i].info : PQ_NULL_INFO);
    }

    inline void insert( // insert item (inlined for speed)
        PQKkey kv,      // key value
        PQKinfo inf)    // item info
    {
        register int i;
        // slide larger values up
        for (i = n; i > 0; i--) {
            if (mk[i - 1].key > kv)
                mk[i] = mk[i - 1];
            else
                break;
        }
        mk[i].key  = kv; // store element here
        mk[i].info = inf;
        if (n < k)
            n++;            // increment number of items
        ANN_FLOP(k - i + 1) // increment floating ops
    }
};

#endif

//----------------------------------------------------------------------
// File:			pr_queue.h
// Programmer:		Sunil Arya and David Mount
// Description:		Include file for priority queue and related
// 					structures.
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

#ifndef PR_QUEUE_H
#define PR_QUEUE_H

//----------------------------------------------------------------------
//	Basic types.
//----------------------------------------------------------------------
typedef void *PQinfo;  // info field is generic pointer
typedef ANNdist PQkey; // key field is distance

//----------------------------------------------------------------------
//	Priority queue
//		A priority queue is a list of items, along with associated
//		priorities.  The basic operations are insert and extract_minimum.
//
//		The priority queue is maintained using a standard binary heap.
//		(Implementation note: Indexing is performed from [1..max] rather
//		than the C standard of [0..max-1].  This simplifies parent/child
//		computations.)  User information consists of a void pointer,
//		and the user is responsible for casting this quantity into whatever
//		useful form is desired.
//
//		Because the priority queue is so central to the efficiency of
//		query processing, all the code is inline.
//----------------------------------------------------------------------

class ANNpr_queue {

    struct pq_node { // node in priority queue
        PQkey key;   // key value
        PQinfo info; // info field
    };
    int n;        // number of items in queue
    int max_size; // maximum queue size
    pq_node *pq;  // the priority queue (array of nodes)

public:
    ANNpr_queue(int max) // constructor (given max size)
    {
        n        = 0;                    // initially empty
        max_size = max;                  // maximum number of items
        pq       = new pq_node[max + 1]; // queue is array [1..max] of nodes
    }

    ~ANNpr_queue() // destructor
    {
        delete[] pq;
    }

    ANNbool empty() // is queue empty?
    {
        if (n == 0)
            return ANNtrue;
        else
            return ANNfalse;
    }

    ANNbool non_empty() // is queue nonempty?
    {
        if (n == 0)
            return ANNfalse;
        else
            return ANNtrue;
    }

    void reset() // make existing queue empty
    {
        n = 0;
    }

    inline void insert( // insert item (inlined for speed)
        PQkey kv,       // key value
        PQinfo inf)     // item info
    {
        if (++n > max_size)
            annError("Priority queue overflow.", ANNabort);
        register int r = n;
        while (r > 1) { // sift up new item
            register int p = r / 2;
            ANN_FLOP(1)          // increment floating ops
            if (pq[p].key <= kv) // in proper order
                break;
            pq[r] = pq[p]; // else swap with parent
            r     = p;
        }
        pq[r].key  = kv; // insert new item at final location
        pq[r].info = inf;
    }

    inline void extr_min( // extract minimum (inlined for speed)
        PQkey &kv,        // key (returned)
        PQinfo &inf)      // item info (returned)
    {
        kv                = pq[1].key;   // key of min item
        inf               = pq[1].info;  // information of min item
        register PQkey kn = pq[n--].key; // last item in queue
        register int p    = 1;           // p points to item out of position
        register int r    = p << 1;      // left child of p
        while (r <= n) {                 // while r is still within the heap
            ANN_FLOP(2)                  // increment floating ops
                                         // set r to smaller child of p
            if (r < n && pq[r].key > pq[r + 1].key)
                r++;
            if (kn <= pq[r].key) // in proper order
                break;
            pq[p] = pq[r]; // else swap with child
            p     = r;     // advance pointers
            r     = p << 1;
        }
        pq[p] = pq[n + 1]; // insert last item in proper place
    }
};

#endif

//----------------------------------------------------------------------
// File:			kd_search.h
// Programmer:		Sunil Arya and David Mount
// Description:		Standard kd-tree search
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

#ifndef ANN_kd_search_H
#define ANN_kd_search_H

//----------------------------------------------------------------------
//	More global variables
//		These are active for the life of each call to annkSearch(). They
//		are set to save the number of variables that need to be passed
//		among the various search procedures.
//----------------------------------------------------------------------
#endif
//----------------------------------------------------------------------
// File:			kd_pr_search.h
// Programmer:		Sunil Arya and David Mount
// Description:		Priority kd-tree search
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

#ifndef ANN_kd_pr_search_H
#define ANN_kd_pr_search_H

//----------------------------------------------------------------------
//	Global variables
//		Active for the life of each call to Appx_Near_Neigh() or
//		Appx_k_Near_Neigh().
//----------------------------------------------------------------------

extern double ANNprEps;         // the error bound
extern int ANNprDim;            // dimension of space
extern ANNpoint ANNprQ;         // query point
extern double ANNprMaxErr;      // max tolerable squared error
extern ANNpointArray ANNprPts;  // the points
extern ANNpr_queue *ANNprBoxPQ; // priority queue for boxes
extern ANNmin_k *ANNprPointMK;  // set of k closest points

#endif
//----------------------------------------------------------------------
// File:			kd_fix_rad_search.h
// Programmer:		Sunil Arya and David Mount
// Description:		Standard kd-tree fixed-radius kNN search
// Last modified:	05/03/05 (Version 1.1)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 1.1  05/03/05
//		Initial release
//----------------------------------------------------------------------

#ifndef ANN_kd_fix_rad_search_H
#define ANN_kd_fix_rad_search_H

//----------------------------------------------------------------------
//	Global variables
//		These are active for the life of each call to
//		annRangeSearch().  They are set to save the number of
//		variables that need to be passed among the various search
//		procedures.
//----------------------------------------------------------------------

extern ANNpoint ANNkdFRQ; // query point (static copy)

#endif

//----------------------------------------------------------------------
// File:			ANN.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Methods for ANN.h and ANNx.h
// Last modified:	01/27/10 (Version 1.1.2)
//----------------------------------------------------------------------
// Copyright (c) 1997-2010 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.0  04/01/05
//		Added performance counting to annDist()
//	Revision 1.1.2  01/27/10
//		Fixed minor compilation bugs for new versions of gcc
//----------------------------------------------------------------------

#include <cstdlib>        // C standard lib defs
                          // make std:: accessible

//----------------------------------------------------------------------
//	Point methods
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Distance utility.
//		(Note: In the nearest neighbor search, most distances are
//		computed using partial distance calculations, not this
//		procedure.)
//----------------------------------------------------------------------

ANNdist annDist( // interpoint squared distance
    int dim, ANNpoint p, ANNpoint q) {
    register int d;
    register ANNcoord diff;
    register ANNcoord dist;

    dist = 0;
    for (d = 0; d < dim; d++) {
        diff = p[d] - q[d];
        dist = ANN_SUM(dist, ANN_POW(diff));
    }
    ANN_FLOP(3 * dim) // performance counts
    ANN_PTS(1)
    ANN_COORD(dim)
    return dist;
}

//----------------------------------------------------------------------
//	annPrintPoint() prints a point to a given output stream.
//----------------------------------------------------------------------

void annPrintPt(       // print a point
    ANNpoint pt,       // the point
    int dim,           // the dimension
    std::ostream &out) // output stream
{
    for (int j = 0; j < dim; j++) {
        out << pt[j];
        if (j < dim - 1)
            out << " ";
    }
}

//----------------------------------------------------------------------
//	Point allocation/deallocation:
//
//		Because points (somewhat like strings in C) are stored
//		as pointers.  Consequently, creating and destroying
//		copies of points may require storage allocation.  These
//		procedures do this.
//
//		annAllocPt() and annDeallocPt() allocate a deallocate
//		storage for a single point, and return a pointer to it.
//
//		annAllocPts() allocates an array of points as well a place
//		to store their coordinates, and initializes the points to
//		point to their respective coordinates.  It allocates point
//		storage in a contiguous block large enough to store all the
//		points.  It performs no initialization.
//
//		annDeallocPts() should only be used on point arrays allocated
//		by annAllocPts since it assumes that points are allocated in
//		a block.
//
//		annCopyPt() copies a point taking care to allocate storage
//		for the new point.
//
//		annAssignRect() assigns the coordinates of one rectangle to
//		another.  The two rectangles must have the same dimension
//		(and it is not possible to test this here).
//----------------------------------------------------------------------

ANNpoint annAllocPt(int dim, ANNcoord c) // allocate 1 point
{
    ANNpoint p = new ANNcoord[dim];
    for (int i = 0; i < dim; i++)
        p[i] = c;
    return p;
}

ANNpointArray annAllocPts(int n, int dim) // allocate n pts in dim
{
    ANNpointArray pa = new ANNpoint[n];       // allocate points
    ANNpoint p       = new ANNcoord[n * dim]; // allocate space for coords
    for (int i = 0; i < n; i++) {
        pa[i] = &(p[i * dim]);
    }
    return pa;
}

void annDeallocPt(ANNpoint &p) // deallocate 1 point
{
    delete[] p;
    p = NULL;
}

void annDeallocPts(ANNpointArray &pa) // deallocate points
{
    delete[] pa[0]; // dealloc coordinate storage
    delete[] pa;    // dealloc points
    pa = NULL;
}

ANNpoint annCopyPt(int dim, ANNpoint source) // copy point
{
    ANNpoint p = new ANNcoord[dim];
    for (int i = 0; i < dim; i++)
        p[i] = source[i];
    return p;
}

// assign one rect to another
void annAssignRect(int dim, ANNorthRect &dest, const ANNorthRect &source) {
    for (int i = 0; i < dim; i++) {
        dest.lo[i] = source.lo[i];
        dest.hi[i] = source.hi[i];
    }
}

// is point inside rectangle?
ANNbool ANNorthRect::inside(int dim, ANNpoint p) {
    for (int i = 0; i < dim; i++) {
        if (p[i] < lo[i] || p[i] > hi[i])
            return ANNfalse;
    }
    return ANNtrue;
}

//----------------------------------------------------------------------
//	Error handler
//----------------------------------------------------------------------

void annError(const char *msg, ANNerr level) {
    if (level == ANNabort) {
        cerr << "ANN: ERROR------->" << msg << "<-------------ERROR\n";
        exit(1);
    } else {
        cerr << "ANN: WARNING----->" << msg << "<-------------WARNING\n";
    }
}

//----------------------------------------------------------------------
//	Limit on number of points visited
//		We have an option for terminating the search early if the
//		number of points visited exceeds some threshold.  If the
//		threshold is 0 (its default)  this means there is no limit
//		and the algorithm applies its normal termination condition.
//		This is for applications where there are real time constraints
//		on the running time of the algorithm.
//----------------------------------------------------------------------

int ANNmaxPtsVisited = 0; // maximum number of pts visited
int ANNptsVisited;        // number of pts visited in search

//----------------------------------------------------------------------
//	Global function declarations
//----------------------------------------------------------------------

void annMaxPtsVisit( // set limit on max. pts to visit in search
    int maxPts)      // the limit
{
    ANNmaxPtsVisited = maxPts;
}

//----------------------------------------------------------------------
// File:			bd_fix_rad_search.cpp
// Programmer:		David Mount
// Description:		Standard bd-tree search
// Last modified:	05/03/05 (Version 1.1)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 1.1  05/03/05
//		Initial release
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Approximate searching for bd-trees.
//		See the file kd_FR_search.cpp for general information on the
//		approximate nearest neighbor search algorithm.  Here we
//		include the extensions for shrinking nodes.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	bd_shrink::ann_FR_search - search a shrinking node
//----------------------------------------------------------------------

void ANNbd_shrink::ann_FR_search(ANNdist box_dist) {
    // check dist calc term cond.
    if (ANNmaxPtsVisited != 0 && ANNptsVisited > ANNmaxPtsVisited)
        return;

    ANNdist inner_dist = 0;            // distance to inner box
    for (int i = 0; i < n_bnds; i++) { // is query point in the box?
        if (bnds[i].out(ANNkdFRQ)) {   // outside this bounding side?
                                       // add to inner distance
            inner_dist = (ANNdist) ANN_SUM(inner_dist, bnds[i].dist(ANNkdFRQ));
        }
    }
    if (inner_dist <= box_dist) {                 // if inner box is closer
        child[ANN_IN]->ann_FR_search(inner_dist); // search inner child first
        child[ANN_OUT]->ann_FR_search(box_dist);  // ...then outer child
    } else {                                      // if outer box is closer
        child[ANN_OUT]->ann_FR_search(box_dist);  // search outer child first
        child[ANN_IN]->ann_FR_search(inner_dist); // ...then outer child
    }
    ANN_FLOP(3 * n_bnds) // increment floating ops
    ANN_SHR(1)           // one more shrinking node
}

//----------------------------------------------------------------------
// File:			bd_pr_search.cpp
// Programmer:		David Mount
// Description:		Priority search for bd-trees
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Approximate priority searching for bd-trees.
//		See the file kd_pr_search.cc for general information on the
//		approximate nearest neighbor priority search algorithm.  Here
//		we include the extensions for shrinking nodes.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	bd_shrink::ann_search - search a shrinking node
//----------------------------------------------------------------------

void ANNbd_shrink::ann_pri_search(ANNdist box_dist) {
    ANNdist inner_dist = 0;            // distance to inner box
    for (int i = 0; i < n_bnds; i++) { // is query point in the box?
        if (bnds[i].out(ANNprQ)) {     // outside this bounding side?
                                       // add to inner distance
            inner_dist = (ANNdist) ANN_SUM(inner_dist, bnds[i].dist(ANNprQ));
        }
    }
    if (inner_dist <= box_dist) {         // if inner box is closer
        if (child[ANN_OUT] != KD_TRIVIAL) // enqueue outer if not trivial
            ANNprBoxPQ->insert(box_dist, child[ANN_OUT]);
        // continue with inner child
        child[ANN_IN]->ann_pri_search(inner_dist);
    } else {                             // if outer box is closer
        if (child[ANN_IN] != KD_TRIVIAL) // enqueue inner if not trivial
            ANNprBoxPQ->insert(inner_dist, child[ANN_IN]);
        // continue with outer child
        child[ANN_OUT]->ann_pri_search(box_dist);
    }
    ANN_FLOP(3 * n_bnds) // increment floating ops
    ANN_SHR(1)           // one more shrinking node
}
//----------------------------------------------------------------------
// File:			bd_search.cpp
// Programmer:		David Mount
// Description:		Standard bd-tree search
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Approximate searching for bd-trees.
//		See the file kd_search.cpp for general information on the
//		approximate nearest neighbor search algorithm.  Here we
//		include the extensions for shrinking nodes.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	bd_shrink::ann_search - search a shrinking node
//----------------------------------------------------------------------

void ANNbd_shrink::ann_search(ANNdist box_dist, ANNkdQueryInfo &qinfo) {
    // check dist calc term cond.
    if (ANNmaxPtsVisited != 0 && ANNptsVisited > ANNmaxPtsVisited)
        return;

    ANNdist inner_dist = 0;              // distance to inner box
    for (int i = 0; i < n_bnds; i++) {   // is query point in the box?
        if (bnds[i].out(qinfo.ANNkdQ)) { // outside this bounding side?
                                         // add to inner distance
            inner_dist = (ANNdist) ANN_SUM(inner_dist, bnds[i].dist(qinfo.ANNkdQ));
        }
    }
    if (inner_dist <= box_dist) {                     // if inner box is closer
        child[ANN_IN]->ann_search(inner_dist, qinfo); // search inner child first
        child[ANN_OUT]->ann_search(box_dist, qinfo);  // ...then outer child
    } else {                                          // if outer box is closer
        child[ANN_OUT]->ann_search(box_dist, qinfo);  // search outer child first
        child[ANN_IN]->ann_search(inner_dist, qinfo); // ...then outer child
    }
    ANN_FLOP(3 * n_bnds) // increment floating ops
    ANN_SHR(1)           // one more shrinking node
}
//----------------------------------------------------------------------
// File:			bd_tree.cpp
// Programmer:		David Mount
// Description:		Basic methods for bd-trees.
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision l.0  04/01/05
//		Fixed centroid shrink threshold condition to depend on the
//			dimension.
//		Moved dump routine to kd_dump.cpp.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Printing a bd-tree
//		These routines print a bd-tree.   See the analogous procedure
//		in kd_tree.cpp for more information.
//----------------------------------------------------------------------

void ANNbd_shrink::print( // print shrinking node
    int level,            // depth of node in tree
    ostream &out)         // output stream
{
    child[ANN_OUT]->print(level + 1, out); // print out-child

    out << "    ";
    for (int i = 0; i < level; i++) // print indentation
        out << "..";
    out << "Shrink";
    for (int j = 0; j < n_bnds; j++) { // print sides, 2 per line
        if (j % 2 == 0) {
            out << "\n"; // newline and indentation
            for (int i = 0; i < level + 2; i++)
                out << "  ";
        }
        out << "  ([" << bnds[j].cd << "]" << (bnds[j].sd > 0 ? ">=" : "< ") << bnds[j].cv << ")";
    }
    out << "\n";

    child[ANN_IN]->print(level + 1, out); // print in-child
}

//----------------------------------------------------------------------
//	kd_tree statistics utility (for performance evaluation)
//		This routine computes various statistics information for
//		shrinking nodes.  See file kd_tree.cpp for more information.
//----------------------------------------------------------------------

void ANNbd_shrink::getStats( // get subtree statistics
    int dim,                 // dimension of space
    ANNkdStats &st,          // stats (modified)
    ANNorthRect &bnd_box)    // bounding box
{
    ANNkdStats ch_stats;        // stats for children
    ANNorthRect inner_box(dim); // inner box of shrink

    annBnds2Box(bnd_box,    // enclosing box
                dim,        // dimension
                n_bnds,     // number of bounds
                bnds,       // bounds array
                inner_box); // inner box (modified)
                            // get stats for inner child
    ch_stats.reset();       // reset
    child[ANN_IN]->getStats(dim, ch_stats, inner_box);
    st.merge(ch_stats); // merge them
                        // get stats for outer child
    ch_stats.reset();   // reset
    child[ANN_OUT]->getStats(dim, ch_stats, bnd_box);
    st.merge(ch_stats); // merge them

    st.depth++; // increment depth
    st.n_shr++; // increment number of shrinks
}

//----------------------------------------------------------------------
// bd-tree constructor
//		This is the main constructor for bd-trees given a set of points.
//		It first builds a skeleton kd-tree as a basis, then computes the
//		bounding box of the data points, and then invokes rbd_tree() to
//		actually build the tree, passing it the appropriate splitting
//		and shrinking information.
//----------------------------------------------------------------------

ANNkd_ptr rbd_tree(          // recursive construction of bd-tree
    ANNpointArray pa,        // point array
    ANNidxArray pidx,        // point indices to store in subtree
    int n,                   // number of points
    int dim,                 // dimension of space
    int bsp,                 // bucket space
    ANNorthRect &bnd_box,    // bounding box for current node
    ANNkd_splitter splitter, // splitting routine
    ANNshrinkRule shrink);   // shrinking rule

ANNbd_tree::ANNbd_tree(     // construct from point array
    ANNpointArray pa,       // point array (with at least n pts)
    int n,                  // number of points
    int dd,                 // dimension
    int bs,                 // bucket size
    ANNsplitRule split,     // splitting rule
    ANNshrinkRule shrink)   // shrinking rule
    : ANNkd_tree(n, dd, bs) // build skeleton base tree
{
    pts = pa; // where the points are
    if (n == 0)
        return; // no points--no sweat

    ANNorthRect bnd_box(dd); // bounding box for points
                             // construct bounding rectangle
    annEnclRect(pa, pidx, n, dd, bnd_box);
    // copy to tree structure
    bnd_box_lo = annCopyPt(dd, bnd_box.lo);
    bnd_box_hi = annCopyPt(dd, bnd_box.hi);

    switch (split) {     // build by rule
        case ANN_KD_STD: // standard kd-splitting rule
            root = rbd_tree(pa, pidx, n, dd, bs, bnd_box, kd_split, shrink);
            break;
        case ANN_KD_MIDPT: // midpoint split
            root = rbd_tree(pa, pidx, n, dd, bs, bnd_box, midpt_split, shrink);
            break;
        case ANN_KD_SUGGEST:  // best (in our opinion)
        case ANN_KD_SL_MIDPT: // sliding midpoint split
            root = rbd_tree(pa, pidx, n, dd, bs, bnd_box, sl_midpt_split, shrink);
            break;
        case ANN_KD_FAIR: // fair split
            root = rbd_tree(pa, pidx, n, dd, bs, bnd_box, fair_split, shrink);
            break;
        case ANN_KD_SL_FAIR: // sliding fair split
            root = rbd_tree(pa, pidx, n, dd, bs, bnd_box, sl_fair_split, shrink);
            break;
        default:
            annError("Illegal splitting method", ANNabort);
    }
}

//----------------------------------------------------------------------
//	Shrinking rules
//----------------------------------------------------------------------

enum ANNdecomp { SPLIT, SHRINK }; // decomposition methods

//----------------------------------------------------------------------
//	trySimpleShrink - Attempt a simple shrink
//
//		We compute the tight bounding box of the points, and compute
//		the 2*dim ``gaps'' between the sides of the tight box and the
//		bounding box.  If any of the gaps is large enough relative to
//		the longest side of the tight bounding box, then we shrink
//		all sides whose gaps are large enough.  (The reason for
//		comparing against the tight bounding box, is that after
//		shrinking the longest box size will decrease, and if we use
//		the standard bounding box, we may decide to shrink twice in
//		a row.  Since the tight box is fixed, we cannot shrink twice
//		consecutively.)
//----------------------------------------------------------------------
const float BD_GAP_THRESH = 0.5; // gap threshold (must be < 1)
const int BD_CT_THRESH    = 2;   // min number of shrink sides

ANNdecomp trySimpleShrink(      // try a simple shrink
    ANNpointArray pa,           // point array
    ANNidxArray pidx,           // point indices to store in subtree
    int n,                      // number of points
    int dim,                    // dimension of space
    const ANNorthRect &bnd_box, // current bounding box
    ANNorthRect &inner_box)     // inner box if shrinking (returned)
{
    int i;
    // compute tight bounding box
    annEnclRect(pa, pidx, n, dim, inner_box);

    ANNcoord max_length = 0; // find longest box side
    for (i = 0; i < dim; i++) {
        ANNcoord length = inner_box.hi[i] - inner_box.lo[i];
        if (length > max_length) {
            max_length = length;
        }
    }

    int shrink_ct = 0;          // number of sides we shrunk
    for (i = 0; i < dim; i++) { // select which sides to shrink
                                // gap between boxes
        ANNcoord gap_hi = bnd_box.hi[i] - inner_box.hi[i];
        // big enough gap to shrink?
        if (gap_hi < max_length * BD_GAP_THRESH)
            inner_box.hi[i] = bnd_box.hi[i]; // no - expand
        else
            shrink_ct++; // yes - shrink this side

        // repeat for high side
        ANNcoord gap_lo = inner_box.lo[i] - bnd_box.lo[i];
        if (gap_lo < max_length * BD_GAP_THRESH)
            inner_box.lo[i] = bnd_box.lo[i]; // no - expand
        else
            shrink_ct++; // yes - shrink this side
    }

    if (shrink_ct >= BD_CT_THRESH) // did we shrink enough sides?
        return SHRINK;
    else
        return SPLIT;
}

//----------------------------------------------------------------------
//	tryCentroidShrink - Attempt a centroid shrink
//
//	We repeatedly apply the splitting rule, always to the larger subset
//	of points, until the number of points decreases by the constant
//	fraction BD_FRACTION.  If this takes more than dim*BD_MAX_SPLIT_FAC
//	splits for this to happen, then we shrink to the final inner box
//	Otherwise we split.
//----------------------------------------------------------------------

const float BD_MAX_SPLIT_FAC = 0.5; // maximum number of splits allowed
const float BD_FRACTION      = 0.5; // ...to reduce points by this fraction
                                    // ...This must be < 1.

ANNdecomp tryCentroidShrink(    // try a centroid shrink
    ANNpointArray pa,           // point array
    ANNidxArray pidx,           // point indices to store in subtree
    int n,                      // number of points
    int dim,                    // dimension of space
    const ANNorthRect &bnd_box, // current bounding box
    ANNkd_splitter splitter,    // splitting procedure
    ANNorthRect &inner_box)     // inner box if shrinking (returned)
{
    int n_sub    = n;                       // number of points in subset
    int n_goal   = (int) (n * BD_FRACTION); // number of point in goal
    int n_splits = 0;                       // number of splits needed
                                            // initialize inner box to bounding box
    annAssignRect(dim, inner_box, bnd_box);

    while (n_sub > n_goal) { // keep splitting until goal reached
        int cd;              // cut dim from splitter (ignored)
        ANNcoord cv;         // cut value from splitter (ignored)
        int n_lo;            // number of points on low side
                             // invoke splitting procedure
        (*splitter)(pa, pidx, inner_box, n_sub, dim, cd, cv, n_lo);
        n_splits++; // increment split count

        if (n_lo >= n_sub / 2) {     // most points on low side
            inner_box.hi[cd] = cv;   // collapse high side
            n_sub            = n_lo; // recurse on lower points
        } else {                     // most points on high side
            inner_box.lo[cd] = cv;   // collapse low side
            pidx += n_lo;            // recurse on higher points
            n_sub -= n_lo;
        }
    }
    if (n_splits > dim * BD_MAX_SPLIT_FAC) // took too many splits
        return SHRINK;                     // shrink to final subset
    else
        return SPLIT;
}

//----------------------------------------------------------------------
//	selectDecomp - select which decomposition to use
//----------------------------------------------------------------------

ANNdecomp selectDecomp(         // select decomposition method
    ANNpointArray pa,           // point array
    ANNidxArray pidx,           // point indices to store in subtree
    int n,                      // number of points
    int dim,                    // dimension of space
    const ANNorthRect &bnd_box, // current bounding box
    ANNkd_splitter splitter,    // splitting procedure
    ANNshrinkRule shrink,       // shrinking rule
    ANNorthRect &inner_box)     // inner box if shrinking (returned)
{
    ANNdecomp decomp = SPLIT; // decomposition

    switch (shrink) {     // check shrinking rule
        case ANN_BD_NONE: // no shrinking allowed
            decomp = SPLIT;
            break;
        case ANN_BD_SUGGEST:                     // author's suggestion
        case ANN_BD_SIMPLE:                      // simple shrink
            decomp = trySimpleShrink(pa, pidx,   // points and indices
                                     n, dim,     // number of points and dimension
                                     bnd_box,    // current bounding box
                                     inner_box); // inner box if shrinking (returned)
            break;
        case ANN_BD_CENTROID:                      // centroid shrink
            decomp = tryCentroidShrink(pa, pidx,   // points and indices
                                       n, dim,     // number of points and dimension
                                       bnd_box,    // current bounding box
                                       splitter,   // splitting procedure
                                       inner_box); // inner box if shrinking (returned)
            break;
        default:
            annError("Illegal shrinking rule", ANNabort);
    }
    return decomp;
}

//----------------------------------------------------------------------
//	rbd_tree - recursive procedure to build a bd-tree
//
//		This is analogous to rkd_tree, but for bd-trees.  See the
//		procedure rkd_tree() in kd_split.cpp for more information.
//
//		If the number of points falls below the bucket size, then a
//		leaf node is created for the points.  Otherwise we invoke the
//		procedure selectDecomp() which determines whether we are to
//		split or shrink.  If splitting is chosen, then we essentially
//		do exactly as rkd_tree() would, and invoke the specified
//		splitting procedure to the points.  Otherwise, the selection
//		procedure returns a bounding box, from which we extract the
//		appropriate shrinking bounds, and create a shrinking node.
//		Finally the points are subdivided, and the procedure is
//		invoked recursively on the two subsets to form the children.
//----------------------------------------------------------------------

ANNkd_ptr rbd_tree(          // recursive construction of bd-tree
    ANNpointArray pa,        // point array
    ANNidxArray pidx,        // point indices to store in subtree
    int n,                   // number of points
    int dim,                 // dimension of space
    int bsp,                 // bucket space
    ANNorthRect &bnd_box,    // bounding box for current node
    ANNkd_splitter splitter, // splitting routine
    ANNshrinkRule shrink)    // shrinking rule
{
    ANNdecomp decomp; // decomposition method

    ANNorthRect inner_box(dim); // inner box (if shrinking)

    if (n <= bsp) {            // n small, make a leaf node
        if (n == 0)            // empty leaf node
            return KD_TRIVIAL; // return (canonical) empty leaf
        else                   // construct the node and return
            return new ANNkd_leaf(n, pidx);
    }

    decomp = selectDecomp( // select decomposition method
        pa, pidx,          // points and indices
        n, dim,            // number of points and dimension
        bnd_box,           // current bounding box
        splitter, shrink,  // splitting/shrinking methods
        inner_box);        // inner box if shrinking (returned)

    if (decomp == SPLIT) { // split selected
        int cd;            // cutting dimension
        ANNcoord cv;       // cutting value
        int n_lo;          // number on low side of cut
                           // invoke splitting procedure
        (*splitter)(pa, pidx, bnd_box, n, dim, cd, cv, n_lo);

        ANNcoord lv = bnd_box.lo[cd]; // save bounds for cutting dimension
        ANNcoord hv = bnd_box.hi[cd];

        bnd_box.hi[cd] = cv;       // modify bounds for left subtree
        ANNkd_ptr lo   = rbd_tree( // build left subtree
            pa, pidx, n_lo,      // ...from pidx[0..n_lo-1]
            dim, bsp, bnd_box, splitter, shrink);
        bnd_box.hi[cd] = hv; // restore bounds

        bnd_box.lo[cd] = cv;             // modify bounds for right subtree
        ANNkd_ptr hi   = rbd_tree(       // build right subtree
            pa, pidx + n_lo, n - n_lo, // ...from pidx[n_lo..n-1]
            dim, bsp, bnd_box, splitter, shrink);
        bnd_box.lo[cd] = lv; // restore bounds
                             // create the splitting node
        return new ANNkd_split(cd, cv, lv, hv, lo, hi);
    } else {        // shrink selected
        int n_in;   // number of points in box
        int n_bnds; // number of bounding sides

        annBoxSplit(   // split points around inner box
            pa,        // points to split
            pidx,      // point indices
            n,         // number of points
            dim,       // dimension
            inner_box, // inner box
            n_in);     // number of points inside (returned)

        ANNkd_ptr in  = rbd_tree( // build inner subtree pidx[0..n_in-1]
            pa, pidx, n_in, dim, bsp, inner_box, splitter, shrink);
        ANNkd_ptr out = rbd_tree( // build outer subtree pidx[n_in..n]
            pa, pidx + n_in, n - n_in, dim, bsp, bnd_box, splitter, shrink);

        ANNorthHSArray bnds = NULL; // bounds (alloc in Box2Bnds and
                                    // ...freed in bd_shrink destroyer)

        annBox2Bnds(   // convert inner box to bounds
            inner_box, // inner box
            bnd_box,   // enclosing box
            dim,       // dimension
            n_bnds,    // number of bounds (returned)
            bnds);     // bounds array (modified)

        // return shrinking node
        return new ANNbd_shrink(n_bnds, bnds, in, out);
    }
}
//----------------------------------------------------------------------
// File:			brute.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Brute-force nearest neighbors
// Last modified:	05/03/05 (Version 1.1)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.1  05/03/05
//		Added fixed-radius kNN search
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//		Brute-force search simply stores a pointer to the list of
//		data points and searches linearly for the nearest neighbor.
//		The k nearest neighbors are stored in a k-element priority
//		queue (which is implemented in a pretty dumb way as well).
//
//		If ANN_ALLOW_SELF_MATCH is ANNfalse then data points at distance
//		zero are not considered.
//
//		Note that the error bound eps is passed in, but it is ignored.
//		These routines compute exact nearest neighbors (which is needed
//		for validation purposes in ann_test.cpp).
//----------------------------------------------------------------------

ANNbruteForce::ANNbruteForce( // constructor from point array
    ANNpointArray pa,         // point array
    int n,                    // number of points
    int dd)                   // dimension
{
    dim   = dd;
    n_pts = n;
    pts   = pa;
}

ANNbruteForce::~ANNbruteForce() {} // destructor (empty)

void ANNbruteForce::annkSearch( // approx k near neighbor search
    ANNpoint q,                 // query point
    int k,                      // number of near neighbors to return
    ANNidxArray nn_idx,         // nearest neighbor indices (returned)
    ANNdistArray dd,            // dist to near neighbors (returned)
    double eps)                 // error bound (ignored)
{
    ANNmin_k mk(k); // construct a k-limited priority queue
    int i;

    if (k > n_pts) { // too many near neighbors?
        annError("Requesting more near neighbors than data points", ANNabort);
    }
    // run every point through queue
    for (i = 0; i < n_pts; i++) {
        // compute distance to point
        ANNdist sqDist = annDist(dim, pts[i], q);
        if (ANN_ALLOW_SELF_MATCH || sqDist != 0)
            mk.insert(sqDist, i);
    }
    for (i = 0; i < k; i++) { // extract the k closest points
        dd[i]     = mk.ith_smallest_key(i);
        nn_idx[i] = mk.ith_smallest_info(i);
    }
}

int ANNbruteForce::annkFRSearch( // approx fixed-radius kNN search
    ANNpoint q,                  // query point
    ANNdist sqRad,               // squared radius
    int k,                       // number of near neighbors to return
    ANNidxArray nn_idx,          // nearest neighbor array (returned)
    ANNdistArray dd,             // dist to near neighbors (returned)
    double eps)                  // error bound
{
    ANNmin_k mk(k); // construct a k-limited priority queue
    int i;
    int pts_in_range = 0; // number of points in query range
                          // run every point through queue
    for (i = 0; i < n_pts; i++) {
        // compute distance to point
        ANNdist sqDist = annDist(dim, pts[i], q);
        if (sqDist <= sqRad &&                       // within radius bound
            (ANN_ALLOW_SELF_MATCH || sqDist != 0)) { // ...and no self match
            mk.insert(sqDist, i);
            pts_in_range++;
        }
    }
    for (i = 0; i < k; i++) { // extract the k closest points
        if (dd != NULL)
            dd[i] = mk.ith_smallest_key(i);
        if (nn_idx != NULL)
            nn_idx[i] = mk.ith_smallest_info(i);
    }

    return pts_in_range;
}
//----------------------------------------------------------------------
// File:			kd_dump.cc
// Programmer:		David Mount
// Description:		Dump and Load for kd- and bd-trees
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.0  04/01/05
//		Moved dump out of kd_tree.cc into this file.
//		Added kd-tree load constructor.
//----------------------------------------------------------------------
// This file contains routines for dumping kd-trees and bd-trees and
// reloading them. (It is an abuse of policy to include both kd- and
// bd-tree routines in the same file, sorry.  There should be no problem
// in deleting the bd- versions of the routines if they are not
// desired.)
//----------------------------------------------------------------------

// make std:: available

//----------------------------------------------------------------------
//		Constants
//----------------------------------------------------------------------

const int STRING_LEN = 500;  // maximum string length
const double EPSILON = 1E-5; // small number for float comparison

enum ANNtreeType { KD_TREE, BD_TREE }; // tree types (used in loading)

//----------------------------------------------------------------------
//		Procedure declarations
//----------------------------------------------------------------------

static ANNkd_ptr annReadDump(  // read dump file
    istream &in,               // input stream
    ANNtreeType tree_type,     // type of tree expected
    ANNpointArray &the_pts,    // new points (if applic)
    ANNidxArray &the_pidx,     // point indices (returned)
    int &the_dim,              // dimension (returned)
    int &the_n_pts,            // number of points (returned)
    int &the_bkt_size,         // bucket size (returned)
    ANNpoint &the_bnd_box_lo,  // low bounding point
    ANNpoint &the_bnd_box_hi); // high bounding point

static ANNkd_ptr annReadTree( // read tree-part of dump file
    istream &in,              // input stream
    ANNtreeType tree_type,    // type of tree expected
    ANNidxArray the_pidx,     // point indices (modified)
    int &next_idx);           // next index (modified)

//----------------------------------------------------------------------
//	ANN kd- and bd-tree Dump Format
//		The dump file begins with a header containing the version of
//		ANN, an optional section containing the points, followed by
//		a description of the tree.	The tree is printed in preorder.
//
//		Format:
//		#ANN <version number> <comments> [END_OF_LINE]
//		points <dim> <n_pts>			(point coordinates: this is optional)
//		0 <xxx> <xxx> ... <xxx>			(point indices and coordinates)
//		1 <xxx> <xxx> ... <xxx>
//		  ...
//		tree <dim> <n_pts> <bkt_size>
//		<xxx> <xxx> ... <xxx>			(lower end of bounding box)
//		<xxx> <xxx> ... <xxx>			(upper end of bounding box)
//				If the tree is null, then a single line "null" is
//				output.	 Otherwise the nodes of the tree are printed
//				one per line in preorder.  Leaves and splitting nodes
//				have the following formats:
//		Leaf node:
//				leaf <n_pts> <bkt[0]> <bkt[1]> ... <bkt[n-1]>
//		Splitting nodes:
//				split <cut_dim> <cut_val> <lo_bound> <hi_bound>
//
//		For bd-trees:
//
//		Shrinking nodes:
//				shrink <n_bnds>
//						<cut_dim> <cut_val> <side>
//						<cut_dim> <cut_val> <side>
//						... (repeated n_bnds times)
//----------------------------------------------------------------------

void ANNkd_tree::Dump( // dump entire tree
    ANNbool with_pts,  // print points as well?
    ostream &out)      // output stream
{
    out << "#ANN " << ANNversion << "\n";
    out.precision(ANNcoordPrec); // use full precision in dumping
    if (with_pts) {              // print point coordinates
        out << "points " << dim << " " << n_pts << "\n";
        for (int i = 0; i < n_pts; i++) {
            out << i << " ";
            annPrintPt(pts[i], dim, out);
            out << "\n";
        }
    }
    out << "tree " // print tree elements
        << dim << " " << n_pts << " " << bkt_size << "\n";

    annPrintPt(bnd_box_lo, dim, out); // print lower bound
    out << "\n";
    annPrintPt(bnd_box_hi, dim, out); // print upper bound
    out << "\n";

    if (root == NULL) // empty tree?
        out << "null\n";
    else {
        root->dump(out); // invoke printing at root
    }
    out.precision(0); // restore default precision
}

void ANNkd_split::dump( // dump a splitting node
    ostream &out)       // output stream
{
    out << "split " << cut_dim << " " << cut_val << " ";
    out << cd_bnds[ANN_LO] << " " << cd_bnds[ANN_HI] << "\n";

    child[ANN_LO]->dump(out); // print low child
    child[ANN_HI]->dump(out); // print high child
}

void ANNkd_leaf::dump( // dump a leaf node
    ostream &out)      // output stream
{
    if (this == KD_TRIVIAL) { // canonical trivial leaf node
        out << "leaf 0\n";    // leaf no points
    } else {
        out << "leaf " << n_pts;
        for (int j = 0; j < n_pts; j++) {
            out << " " << bkt[j];
        }
        out << "\n";
    }
}

void ANNbd_shrink::dump( // dump a shrinking node
    ostream &out)        // output stream
{
    out << "shrink " << n_bnds << "\n";
    for (int j = 0; j < n_bnds; j++) {
        out << bnds[j].cd << " " << bnds[j].cv << " " << bnds[j].sd << "\n";
    }
    child[ANN_IN]->dump(out);  // print in-child
    child[ANN_OUT]->dump(out); // print out-child
}

//----------------------------------------------------------------------
// Load kd-tree from dump file
//		This rebuilds a kd-tree which was dumped to a file.	 The dump
//		file contains all the basic tree information according to a
//		preorder traversal.	 We assume that the dump file also contains
//		point data.	 (This is to guarantee the consistency of the tree.)
//		If not, then an error is generated.
//
//		Indirectly, this procedure allocates space for points, point
//		indices, all nodes in the tree, and the bounding box for the
//		tree.  When the tree is destroyed, all but the points are
//		deallocated.
//
//		This routine calls annReadDump to do all the work.
//----------------------------------------------------------------------

ANNkd_tree::ANNkd_tree( // build from dump file
    istream &in)        // input stream for dump file
{
    int the_dim;             // local dimension
    int the_n_pts;           // local number of points
    int the_bkt_size;        // local number of points
    ANNpoint the_bnd_box_lo; // low bounding point
    ANNpoint the_bnd_box_hi; // high bounding point
    ANNpointArray the_pts;   // point storage
    ANNidxArray the_pidx;    // point index storage
    ANNkd_ptr the_root;      // root of the tree

    the_root = annReadDump(               // read the dump file
        in,                               // input stream
        KD_TREE,                          // expecting a kd-tree
        the_pts,                          // point array (returned)
        the_pidx,                         // point indices (returned)
        the_dim, the_n_pts, the_bkt_size, // basic tree info (returned)
        the_bnd_box_lo, the_bnd_box_hi);  // bounding box info (returned)

    // create a skeletal tree
    SkeletonTree(the_n_pts, the_dim, the_bkt_size, the_pts, the_pidx);

    bnd_box_lo = the_bnd_box_lo;
    bnd_box_hi = the_bnd_box_hi;

    root = the_root; // set the root
}

ANNbd_tree::ANNbd_tree( // build bd-tree from dump file
    istream &in)
    : ANNkd_tree() // input stream for dump file
{
    int the_dim;             // local dimension
    int the_n_pts;           // local number of points
    int the_bkt_size;        // local number of points
    ANNpoint the_bnd_box_lo; // low bounding point
    ANNpoint the_bnd_box_hi; // high bounding point
    ANNpointArray the_pts;   // point storage
    ANNidxArray the_pidx;    // point index storage
    ANNkd_ptr the_root;      // root of the tree

    the_root = annReadDump(               // read the dump file
        in,                               // input stream
        BD_TREE,                          // expecting a bd-tree
        the_pts,                          // point array (returned)
        the_pidx,                         // point indices (returned)
        the_dim, the_n_pts, the_bkt_size, // basic tree info (returned)
        the_bnd_box_lo, the_bnd_box_hi);  // bounding box info (returned)

    // create a skeletal tree
    SkeletonTree(the_n_pts, the_dim, the_bkt_size, the_pts, the_pidx);
    bnd_box_lo = the_bnd_box_lo;
    bnd_box_hi = the_bnd_box_hi;

    root = the_root; // set the root
}

//----------------------------------------------------------------------
//	annReadDump - read a dump file
//
//		This procedure reads a dump file, constructs a kd-tree
//		and returns all the essential information needed to actually
//		construct the tree.	 Because this procedure is used for
//		constructing both kd-trees and bd-trees, the second argument
//		is used to indicate which type of tree we are expecting.
//----------------------------------------------------------------------

static ANNkd_ptr annReadDump(istream &in,              // input stream
                             ANNtreeType tree_type,    // type of tree expected
                             ANNpointArray &the_pts,   // new points (returned)
                             ANNidxArray &the_pidx,    // point indices (returned)
                             int &the_dim,             // dimension (returned)
                             int &the_n_pts,           // number of points (returned)
                             int &the_bkt_size,        // bucket size (returned)
                             ANNpoint &the_bnd_box_lo, // low bounding point (ret'd)
                             ANNpoint &the_bnd_box_hi) // high bounding point (ret'd)
{
    int j;
    char str[STRING_LEN];     // storage for string
    char version[STRING_LEN]; // ANN version number
    ANNkd_ptr the_root = NULL;

    //------------------------------------------------------------------
    //	Input file header
    //------------------------------------------------------------------
    in >> str;                      // input header
    if (strcmp(str, "#ANN") != 0) { // incorrect header
        annError("Incorrect header for dump file", ANNabort);
    }
    in.getline(version, STRING_LEN); // get version (ignore)

    //------------------------------------------------------------------
    //	Input the points
    //			An array the_pts is allocated and points are read from
    //			the dump file.
    //------------------------------------------------------------------
    in >> str;                        // get major heading
    if (strcmp(str, "points") == 0) { // points section
        in >> the_dim;                // input dimension
        in >> the_n_pts;              // number of points
                                      // allocate point storage
        the_pts = annAllocPts(the_n_pts, the_dim);
        for (int i = 0; i < the_n_pts; i++) { // input point coordinates
            ANNidx idx;                       // point index
            in >> idx;                        // input point index
            if (idx < 0 || idx >= the_n_pts) {
                annError("Point index is out of range", ANNabort);
            }
            for (j = 0; j < the_dim; j++) {
                in >> the_pts[idx][j]; // read point coordinates
            }
        }
        in >> str; // get next major heading
    } else {       // no points were input
        annError("Points must be supplied in the dump file", ANNabort);
    }

    //------------------------------------------------------------------
    //	Input the tree
    //			After the basic header information, we invoke annReadTree
    //			to do all the heavy work.  We create our own array of
    //			point indices (so we can pass them to annReadTree())
    //			but we do not deallocate them.	They will be deallocated
    //			when the tree is destroyed.
    //------------------------------------------------------------------
    if (strcmp(str, "tree") == 0) {           // tree section
        in >> the_dim;                        // read dimension
        in >> the_n_pts;                      // number of points
        in >> the_bkt_size;                   // bucket size
        the_bnd_box_lo = annAllocPt(the_dim); // allocate bounding box pts
        the_bnd_box_hi = annAllocPt(the_dim);

        for (j = 0; j < the_dim; j++) { // read bounding box low
            in >> the_bnd_box_lo[j];
        }
        for (j = 0; j < the_dim; j++) { // read bounding box low
            in >> the_bnd_box_hi[j];
        }
        the_pidx     = new ANNidx[the_n_pts]; // allocate point index array
        int next_idx = 0;                     // number of indices filled
                                              // read the tree and indices
        the_root = annReadTree(in, tree_type, the_pidx, next_idx);
        if (next_idx != the_n_pts) { // didn't see all the points?
            annError("Didn't see as many points as expected", ANNwarn);
        }
    } else {
        annError("Illegal dump format.	Expecting section heading", ANNabort);
    }
    return the_root;
}

//----------------------------------------------------------------------
// annReadTree - input tree and return pointer
//
//		annReadTree reads in a node of the tree, makes any recursive
//		calls as needed to input the children of this node (if internal).
//		It returns a pointer to the node that was created.	An array
//		of point indices is given along with a pointer to the next
//		available location in the array.  As leaves are read, their
//		point indices are stored here, and the point buckets point
//		to the first entry in the array.
//
//		Recall that these are the formats.	The tree is given in
//		preorder.
//
//		Leaf node:
//				leaf <n_pts> <bkt[0]> <bkt[1]> ... <bkt[n-1]>
//		Splitting nodes:
//				split <cut_dim> <cut_val> <lo_bound> <hi_bound>
//
//		For bd-trees:
//
//		Shrinking nodes:
//				shrink <n_bnds>
//						<cut_dim> <cut_val> <side>
//						<cut_dim> <cut_val> <side>
//						... (repeated n_bnds times)
//----------------------------------------------------------------------

static ANNkd_ptr annReadTree(istream &in,           // input stream
                             ANNtreeType tree_type, // type of tree expected
                             ANNidxArray the_pidx,  // point indices (modified)
                             int &next_idx)         // next index (modified)
{
    char tag[STRING_LEN]; // tag (leaf, split, shrink)
    int n_pts;            // number of points in leaf
    int cd;               // cut dimension
    ANNcoord cv;          // cut value
    ANNcoord lb;          // low bound
    ANNcoord hb;          // high bound
    int n_bnds;           // number of bounding sides
    int sd;               // which side

    in >> tag; // input node tag

    if (strcmp(tag, "null") == 0) { // null tree
        return NULL;
    }
    //------------------------------------------------------------------
    //	Read a leaf
    //------------------------------------------------------------------
    if (strcmp(tag, "leaf") == 0) { // leaf node

        in >> n_pts;            // input number of points
        int old_idx = next_idx; // save next_idx
        if (n_pts == 0) {       // trivial leaf
            return KD_TRIVIAL;
        } else {
            for (int i = 0; i < n_pts; i++) { // input point indices
                in >> the_pidx[next_idx++];   // store in array of indices
            }
        }
        return new ANNkd_leaf(n_pts, &the_pidx[old_idx]);
    }
    //------------------------------------------------------------------
    //	Read a splitting node
    //------------------------------------------------------------------
    else if (strcmp(tag, "split") == 0) { // splitting node

        in >> cd >> cv >> lb >> hb;

        // read low and high subtrees
        ANNkd_ptr lc = annReadTree(in, tree_type, the_pidx, next_idx);
        ANNkd_ptr hc = annReadTree(in, tree_type, the_pidx, next_idx);
        // create new node and return
        return new ANNkd_split(cd, cv, lb, hb, lc, hc);
    }
    //------------------------------------------------------------------
    //	Read a shrinking node (bd-tree only)
    //------------------------------------------------------------------
    else if (strcmp(tag, "shrink") == 0) { // shrinking node
        if (tree_type != BD_TREE) {
            annError("Shrinking node not allowed in kd-tree", ANNabort);
        }

        in >> n_bnds; // number of bounding sides
                      // allocate bounds array
        ANNorthHSArray bds = new ANNorthHalfSpace[n_bnds];
        for (int i = 0; i < n_bnds; i++) {
            in >> cd >> cv >> sd; // input bounding halfspace
                                  // copy to array
            bds[i] = ANNorthHalfSpace(cd, cv, sd);
        }
        // read inner and outer subtrees
        ANNkd_ptr ic = annReadTree(in, tree_type, the_pidx, next_idx);
        ANNkd_ptr oc = annReadTree(in, tree_type, the_pidx, next_idx);
        // create new node and return
        return new ANNbd_shrink(n_bnds, bds, ic, oc);
    } else {
        annError("Illegal node type in dump file", ANNabort);
        exit(0); // to keep the compiler happy
    }
}
//----------------------------------------------------------------------
// File:			kd_fix_rad_search.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Standard kd-tree fixed-radius kNN search
// Last modified:	05/03/05 (Version 1.1)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 1.1  05/03/05
//		Initial release
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Approximate fixed-radius k nearest neighbor search
//		The squared radius is provided, and this procedure finds the
//		k nearest neighbors within the radius, and returns the total
//		number of points lying within the radius.
//
//		The method used for searching the kd-tree is a variation of the
//		nearest neighbor search used in kd_search.cpp, except that the
//		radius of the search ball is known.  We refer the reader to that
//		file for the explanation of the recursive search procedure.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//		To keep argument lists short, a number of global variables
//		are maintained which are common to all the recursive calls.
//		These are given below.
//----------------------------------------------------------------------

int ANNkdFRDim;           // dimension of space
ANNpoint ANNkdFRQ;        // query point
ANNdist ANNkdFRSqRad;     // squared radius search bound
double ANNkdFRMaxErr;     // max tolerable squared error
ANNpointArray ANNkdFRPts; // the points
ANNmin_k *ANNkdFRPointMK; // set of k closest points
int ANNkdFRPtsVisited;    // total points visited
int ANNkdFRPtsInRange;    // number of points in the range

//----------------------------------------------------------------------
//	annkFRSearch - fixed radius search for k nearest neighbors
//----------------------------------------------------------------------

int ANNkd_tree::annkFRSearch(ANNpoint q,         // the query point
                             ANNdist sqRad,      // squared radius search bound
                             int k,              // number of near neighbors to return
                             ANNidxArray nn_idx, // nearest neighbor indices (returned)
                             ANNdistArray dd,    // the approximate nearest neighbor
                             double eps)         // the error bound
{
    ANNkdFRDim        = dim; // copy arguments to static equivs
    ANNkdFRQ          = q;
    ANNkdFRSqRad      = sqRad;
    ANNkdFRPts        = pts;
    ANNkdFRPtsVisited = 0; // initialize count of points visited
    ANNkdFRPtsInRange = 0; // ...and points in the range

    ANNkdFRMaxErr = ANN_POW(1.0 + eps);
    ANN_FLOP(2) // increment floating op count

    ANNkdFRPointMK = new ANNmin_k(k); // create set for closest k points
                                      // search starting at the root
    root->ann_FR_search(annBoxDistance(q, bnd_box_lo, bnd_box_hi, dim));

    for (int i = 0; i < k; i++) { // extract the k-th closest points
        if (dd != NULL)
            dd[i] = ANNkdFRPointMK->ith_smallest_key(i);
        if (nn_idx != NULL)
            nn_idx[i] = ANNkdFRPointMK->ith_smallest_info(i);
    }

    delete ANNkdFRPointMK;    // deallocate closest point set
    return ANNkdFRPtsInRange; // return final point count
}

//----------------------------------------------------------------------
//	kd_split::ann_FR_search - search a splitting node
//		Note: This routine is similar in structure to the standard kNN
//		search.  It visits the subtree that is closer to the query point
//		first.  For fixed-radius search, there is no benefit in visiting
//		one subtree before the other, but we maintain the same basic
//		code structure for the sake of uniformity.
//----------------------------------------------------------------------

void ANNkd_split::ann_FR_search(ANNdist box_dist) {
    // check dist calc term condition
    if (ANNmaxPtsVisited != 0 && ANNkdFRPtsVisited > ANNmaxPtsVisited)
        return;

    // distance to cutting plane
    ANNcoord cut_diff = ANNkdFRQ[cut_dim] - cut_val;

    if (cut_diff < 0) {                         // left of cutting plane
        child[ANN_LO]->ann_FR_search(box_dist); // visit closer child first

        ANNcoord box_diff = cd_bnds[ANN_LO] - ANNkdFRQ[cut_dim];
        if (box_diff < 0) // within bounds - ignore
            box_diff = 0;
        // distance to further box
        box_dist = (ANNdist) ANN_SUM(box_dist, ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

        // visit further child if in range
        if (box_dist * ANNkdFRMaxErr <= ANNkdFRSqRad)
            child[ANN_HI]->ann_FR_search(box_dist);

    } else {                                    // right of cutting plane
        child[ANN_HI]->ann_FR_search(box_dist); // visit closer child first

        ANNcoord box_diff = ANNkdFRQ[cut_dim] - cd_bnds[ANN_HI];
        if (box_diff < 0) // within bounds - ignore
            box_diff = 0;
        // distance to further box
        box_dist = (ANNdist) ANN_SUM(box_dist, ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

        // visit further child if close enough
        if (box_dist * ANNkdFRMaxErr <= ANNkdFRSqRad)
            child[ANN_LO]->ann_FR_search(box_dist);
    }
    ANN_FLOP(13) // increment floating ops
    ANN_SPL(1)   // one more splitting node visited
}

//----------------------------------------------------------------------
//	kd_leaf::ann_FR_search - search points in a leaf node
//		Note: The unreadability of this code is the result of
//		some fine tuning to replace indexing by pointer operations.
//----------------------------------------------------------------------

void ANNkd_leaf::ann_FR_search(ANNdist box_dist) {
    register ANNdist dist; // distance to data point
    register ANNcoord *pp; // data coordinate pointer
    register ANNcoord *qq; // query coordinate pointer
    register ANNcoord t;
    register int d;

    for (int i = 0; i < n_pts; i++) { // check points in bucket

        pp   = ANNkdFRPts[bkt[i]]; // first coord of next data point
        qq   = ANNkdFRQ;           // first coord of query point
        dist = 0;

        for (d = 0; d < ANNkdFRDim; d++) {
            ANN_COORD(1) // one more coordinate hit
            ANN_FLOP(5)  // increment floating ops

            t = *(qq++) - *(pp++); // compute length and adv coordinate
                                   // exceeds dist to k-th smallest?
            if ((dist = ANN_SUM(dist, ANN_POW(t))) > ANNkdFRSqRad) {
                break;
            }
        }

        if (d >= ANNkdFRDim &&                     // among the k best?
            (ANN_ALLOW_SELF_MATCH || dist != 0)) { // and no self-match problem
                                                   // add it to the list
            ANNkdFRPointMK->insert(dist, bkt[i]);
            ANNkdFRPtsInRange++; // increment point count
        }
    }
    ANN_LEAF(1)                 // one more leaf node visited
    ANN_PTS(n_pts)              // increment points visited
    ANNkdFRPtsVisited += n_pts; // increment number of points visited
}
//----------------------------------------------------------------------
// File:			kd_pr_search.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Priority search for kd-trees
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Approximate nearest neighbor searching by priority search.
//		The kd-tree is searched for an approximate nearest neighbor.
//		The point is returned through one of the arguments, and the
//		distance returned is the SQUARED distance to this point.
//
//		The method used for searching the kd-tree is called priority
//		search.  (It is described in Arya and Mount, ``Algorithms for
//		fast vector quantization,'' Proc. of DCC '93: Data Compression
//		Conference}, eds. J. A. Storer and M. Cohn, IEEE Press, 1993,
//		381--390.)
//
//		The cell of the kd-tree containing the query point is located,
//		and cells are visited in increasing order of distance from the
//		query point.  This is done by placing each subtree which has
//		NOT been visited in a priority queue, according to the closest
//		distance of the corresponding enclosing rectangle from the
//		query point.  The search stops when the distance to the nearest
//		remaining rectangle exceeds the distance to the nearest point
//		seen by a factor of more than 1/(1+eps). (Implying that any
//		point found subsequently in the search cannot be closer by more
//		than this factor.)
//
//		The main entry point is annkPriSearch() which sets things up and
//		then call the recursive routine ann_pri_search().  This is a
//		recursive routine which performs the processing for one node in
//		the kd-tree.  There are two versions of this virtual procedure,
//		one for splitting nodes and one for leaves. When a splitting node
//		is visited, we determine which child to continue the search on
//		(the closer one), and insert the other child into the priority
//		queue.  When a leaf is visited, we compute the distances to the
//		points in the buckets, and update information on the closest
//		points.
//
//		Some trickery is used to incrementally update the distance from
//		a kd-tree rectangle to the query point.  This comes about from
//		the fact that which each successive split, only one component
//		(along the dimension that is split) of the squared distance to
//		the child rectangle is different from the squared distance to
//		the parent rectangle.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//		To keep argument lists short, a number of global variables
//		are maintained which are common to all the recursive calls.
//		These are given below.
//----------------------------------------------------------------------

double ANNprEps;         // the error bound
int ANNprDim;            // dimension of space
ANNpoint ANNprQ;         // query point
double ANNprMaxErr;      // max tolerable squared error
ANNpointArray ANNprPts;  // the points
ANNpr_queue *ANNprBoxPQ; // priority queue for boxes
ANNmin_k *ANNprPointMK;  // set of k closest points

//----------------------------------------------------------------------
//	annkPriSearch - priority search for k nearest neighbors
//----------------------------------------------------------------------

void ANNkd_tree::annkPriSearch(ANNpoint q,         // query point
                               int k,              // number of near neighbors to return
                               ANNidxArray nn_idx, // nearest neighbor indices (returned)
                               ANNdistArray dd,    // dist to near neighbors (returned)
                               double eps)         // error bound (ignored)
{
    // max tolerable squared error
    ANNprMaxErr = ANN_POW(1.0 + eps);
    ANN_FLOP(2) // increment floating ops

    ANNprDim      = dim; // copy arguments to static equivs
    ANNprQ        = q;
    ANNprPts      = pts;
    ANNptsVisited = 0; // initialize count of points visited

    ANNprPointMK = new ANNmin_k(k); // create set for closest k points

    // distance to root box
    ANNdist box_dist = annBoxDistance(q, bnd_box_lo, bnd_box_hi, dim);

    ANNprBoxPQ = new ANNpr_queue(n_pts); // create priority queue for boxes
    ANNprBoxPQ->insert(box_dist, root);  // insert root in priority queue

    while (ANNprBoxPQ->non_empty() && (!(ANNmaxPtsVisited != 0 && ANNptsVisited > ANNmaxPtsVisited))) {
        ANNkd_ptr np; // next box from prior queue

        // extract closest box from queue
        ANNprBoxPQ->extr_min(box_dist, (void *&) np);

        ANN_FLOP(2) // increment floating ops
        if (box_dist * ANNprMaxErr >= ANNprPointMK->max_key())
            break;

        np->ann_pri_search(box_dist); // search this subtree.
    }

    for (int i = 0; i < k; i++) { // extract the k-th closest points
        dd[i]     = ANNprPointMK->ith_smallest_key(i);
        nn_idx[i] = ANNprPointMK->ith_smallest_info(i);
    }

    delete ANNprPointMK; // deallocate closest point set
    delete ANNprBoxPQ;   // deallocate priority queue
}

//----------------------------------------------------------------------
//	kd_split::ann_pri_search - search a splitting node
//----------------------------------------------------------------------

void ANNkd_split::ann_pri_search(ANNdist box_dist) {
    ANNdist new_dist; // distance to child visited later
                      // distance to cutting plane
    ANNcoord cut_diff = ANNprQ[cut_dim] - cut_val;

    if (cut_diff < 0) { // left of cutting plane
        ANNcoord box_diff = cd_bnds[ANN_LO] - ANNprQ[cut_dim];
        if (box_diff < 0) // within bounds - ignore
            box_diff = 0;
        // distance to further box
        new_dist = (ANNdist) ANN_SUM(box_dist, ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

        if (child[ANN_HI] != KD_TRIVIAL) // enqueue if not trivial
            ANNprBoxPQ->insert(new_dist, child[ANN_HI]);
        // continue with closer child
        child[ANN_LO]->ann_pri_search(box_dist);
    } else { // right of cutting plane
        ANNcoord box_diff = ANNprQ[cut_dim] - cd_bnds[ANN_HI];
        if (box_diff < 0) // within bounds - ignore
            box_diff = 0;
        // distance to further box
        new_dist = (ANNdist) ANN_SUM(box_dist, ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

        if (child[ANN_LO] != KD_TRIVIAL) // enqueue if not trivial
            ANNprBoxPQ->insert(new_dist, child[ANN_LO]);
        // continue with closer child
        child[ANN_HI]->ann_pri_search(box_dist);
    }
    ANN_SPL(1)  // one more splitting node visited
    ANN_FLOP(8) // increment floating ops
}

//----------------------------------------------------------------------
//	kd_leaf::ann_pri_search - search points in a leaf node
//
//		This is virtually identical to the ann_search for standard search.
//----------------------------------------------------------------------

void ANNkd_leaf::ann_pri_search(ANNdist box_dist) {
    register ANNdist dist;     // distance to data point
    register ANNcoord *pp;     // data coordinate pointer
    register ANNcoord *qq;     // query coordinate pointer
    register ANNdist min_dist; // distance to k-th closest point
    register ANNcoord t;
    register int d;

    min_dist = ANNprPointMK->max_key(); // k-th smallest distance so far

    for (int i = 0; i < n_pts; i++) { // check points in bucket

        pp   = ANNprPts[bkt[i]]; // first coord of next data point
        qq   = ANNprQ;           // first coord of query point
        dist = 0;

        for (d = 0; d < ANNprDim; d++) {
            ANN_COORD(1) // one more coordinate hit
            ANN_FLOP(4)  // increment floating ops

            t = *(qq++) - *(pp++); // compute length and adv coordinate
                                   // exceeds dist to k-th smallest?
            if ((dist = ANN_SUM(dist, ANN_POW(t))) > min_dist) {
                break;
            }
        }

        if (d >= ANNprDim &&                       // among the k best?
            (ANN_ALLOW_SELF_MATCH || dist != 0)) { // and no self-match problem
                                                   // add it to the list
            ANNprPointMK->insert(dist, bkt[i]);
            min_dist = ANNprPointMK->max_key();
        }
    }
    ANN_LEAF(1)             // one more leaf node visited
    ANN_PTS(n_pts)          // increment points visited
    ANNptsVisited += n_pts; // increment number of points visited
}
//----------------------------------------------------------------------
// File:			kd_search.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Standard kd-tree search
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.0  04/01/05
//		Changed names LO, HI to ANN_LO, ANN_HI
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Approximate nearest neighbor searching by kd-tree search
//		The kd-tree is searched for an approximate nearest neighbor.
//		The point is returned through one of the arguments, and the
//		distance returned is the squared distance to this point.
//
//		The method used for searching the kd-tree is an approximate
//		adaptation of the search algorithm described by Friedman,
//		Bentley, and Finkel, ``An algorithm for finding best matches
//		in logarithmic expected time,'' ACM Transactions on Mathematical
//		Software, 3(3):209-226, 1977).
//
//		The algorithm operates recursively.  When first encountering a
//		node of the kd-tree we first visit the child which is closest to
//		the query point.  On return, we decide whether we want to visit
//		the other child.  If the box containing the other child exceeds
//		1/(1+eps) times the current best distance, then we skip it (since
//		any point found in this child cannot be closer to the query point
//		by more than this factor.)  Otherwise, we visit it recursively.
//		The distance between a box and the query point is computed exactly
//		(not approximated as is often done in kd-tree), using incremental
//		distance updates, as described by Arya and Mount in ``Algorithms
//		for fast vector quantization,'' Proc.  of DCC '93: Data Compression
//		Conference, eds. J. A. Storer and M. Cohn, IEEE Press, 1993,
//		381-390.
//
//		The main entry points is annkSearch() which sets things up and
//		then call the recursive routine ann_search().  This is a recursive
//		routine which performs the processing for one node in the kd-tree.
//		There are two versions of this virtual procedure, one for splitting
//		nodes and one for leaves.  When a splitting node is visited, we
//		determine which child to visit first (the closer one), and visit
//		the other child on return.  When a leaf is visited, we compute
//		the distances to the points in the buckets, and update information
//		on the closest points.
//
//		Some trickery is used to incrementally update the distance from
//		a kd-tree rectangle to the query point.  This comes about from
//		the fact that which each successive split, only one component
//		(along the dimension that is split) of the squared distance to
//		the child rectangle is different from the squared distance to
//		the parent rectangle.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//		To keep argument lists short, a number of global variables
//		are maintained which are common to all the recursive calls.
//		These are given below.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	annkSearch - search for the k nearest neighbors
//----------------------------------------------------------------------

void ANNkd_tree::annkSearch(ANNpoint q,         // the query point
                            int k,              // number of near neighbors to return
                            ANNidxArray nn_idx, // nearest neighbor indices (returned)
                            ANNdistArray dd,    // the approximate nearest neighbor
                            double eps)         // the error bound
{
    ANNkdQueryInfo qinfo;
    qinfo.ANNkdDim = dim; // copy arguments to static equivs
    qinfo.ANNkdQ   = q;
    qinfo.ANNkdPts = pts;
    ANNptsVisited  = 0; // initialize count of points visited

    if (k > n_pts) { // too many near neighbors?
        annError("Requesting more near neighbors than data points", ANNabort);
    }

    qinfo.ANNkdMaxErr = ANN_POW(1.0 + eps);
    ANN_FLOP(2) // increment floating op count

    qinfo.ANNkdPointMK = new ANNmin_k(k); // create set for closest k points
                                          // search starting at the root
    root->ann_search(annBoxDistance(q, bnd_box_lo, bnd_box_hi, dim), qinfo);

    for (int i = 0; i < k; i++) { // extract the k-th closest points
        dd[i]     = qinfo.ANNkdPointMK->ith_smallest_key(i);
        nn_idx[i] = qinfo.ANNkdPointMK->ith_smallest_info(i);
    }
    delete qinfo.ANNkdPointMK; // deallocate closest point set
}

//----------------------------------------------------------------------
//	kd_split::ann_search - search a splitting node
//----------------------------------------------------------------------

void ANNkd_split::ann_search(ANNdist box_dist, ANNkdQueryInfo &qinfo) {
    // check dist calc term condition
    if (ANNmaxPtsVisited != 0 && ANNptsVisited > ANNmaxPtsVisited)
        return;

    // distance to cutting plane
    ANNcoord cut_diff = qinfo.ANNkdQ[cut_dim] - cut_val;

    if (cut_diff < 0) {                             // left of cutting plane
        child[ANN_LO]->ann_search(box_dist, qinfo); // visit closer child first

        ANNcoord box_diff = cd_bnds[ANN_LO] - qinfo.ANNkdQ[cut_dim];
        if (box_diff < 0) // within bounds - ignore
            box_diff = 0;
        // distance to further box
        box_dist = (ANNdist) ANN_SUM(box_dist, ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

        // visit further child if close enough
        if (box_dist * qinfo.ANNkdMaxErr < qinfo.ANNkdPointMK->max_key())
            child[ANN_HI]->ann_search(box_dist, qinfo);

    } else {                                        // right of cutting plane
        child[ANN_HI]->ann_search(box_dist, qinfo); // visit closer child first

        ANNcoord box_diff = qinfo.ANNkdQ[cut_dim] - cd_bnds[ANN_HI];
        if (box_diff < 0) // within bounds - ignore
            box_diff = 0;
        // distance to further box
        box_dist = (ANNdist) ANN_SUM(box_dist, ANN_DIFF(ANN_POW(box_diff), ANN_POW(cut_diff)));

        // visit further child if close enough
        if (box_dist * qinfo.ANNkdMaxErr < qinfo.ANNkdPointMK->max_key())
            child[ANN_LO]->ann_search(box_dist, qinfo);
    }
    ANN_FLOP(10) // increment floating ops
    ANN_SPL(1)   // one more splitting node visited
}

//----------------------------------------------------------------------
//	kd_leaf::ann_search - search points in a leaf node
//		Note: The unreadability of this code is the result of
//		some fine tuning to replace indexing by pointer operations.
//----------------------------------------------------------------------

void ANNkd_leaf::ann_search(ANNdist box_dist, ANNkdQueryInfo &qinfo) {
    register ANNdist dist;     // distance to data point
    register ANNcoord *pp;     // data coordinate pointer
    register ANNcoord *qq;     // query coordinate pointer
    register ANNdist min_dist; // distance to k-th closest point
    register ANNcoord t;
    register int d;

    min_dist = qinfo.ANNkdPointMK->max_key(); // k-th smallest distance so far

    for (int i = 0; i < n_pts; i++) { // check points in bucket

        pp   = qinfo.ANNkdPts[bkt[i]]; // first coord of next data point
        qq   = qinfo.ANNkdQ;           // first coord of query point
        dist = 0;

        for (d = 0; d < qinfo.ANNkdDim; d++) {
            ANN_COORD(1) // one more coordinate hit
            ANN_FLOP(4)  // increment floating ops

            t = *(qq++) - *(pp++); // compute length and adv coordinate
                                   // exceeds dist to k-th smallest?
            if ((dist = ANN_SUM(dist, ANN_POW(t))) > min_dist) {
                break;
            }
        }

        if (d >= qinfo.ANNkdDim &&                 // among the k best?
            (ANN_ALLOW_SELF_MATCH || dist != 0)) { // and no self-match problem
                                                   // add it to the list
            qinfo.ANNkdPointMK->insert(dist, bkt[i]);
            min_dist = qinfo.ANNkdPointMK->max_key();
        }
    }
    ANN_LEAF(1)             // one more leaf node visited
    ANN_PTS(n_pts)          // increment points visited
    ANNptsVisited += n_pts; // increment number of points visited
}
//----------------------------------------------------------------------
// File:			kd_split.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Methods for splitting kd-trees
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.0  04/01/05
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Constants
//----------------------------------------------------------------------

const double ERR             = 0.001; // a small value
const double FS_ASPECT_RATIO = 3.0;   // maximum allowed aspect ratio
                                      // in fair split. Must be >= 2.

//----------------------------------------------------------------------
//	kd_split - Bentley's standard splitting routine for kd-trees
//		Find the dimension of the greatest spread, and split
//		just before the median point along this dimension.
//----------------------------------------------------------------------

void kd_split(ANNpointArray pa,        // point array (permuted on return)
              ANNidxArray pidx,        // point indices
              const ANNorthRect &bnds, // bounding rectangle for cell
              int n,                   // number of points
              int dim,                 // dimension of space
              int &cut_dim,            // cutting dimension (returned)
              ANNcoord &cut_val,       // cutting value (returned)
              int &n_lo)               // num of points on low side (returned)
{
    // find dimension of maximum spread
    cut_dim = annMaxSpread(pa, pidx, n, dim);
    n_lo    = n / 2; // median rank
                     // split about median
    annMedianSplit(pa, pidx, n, cut_dim, cut_val, n_lo);
}

//----------------------------------------------------------------------
//	midpt_split - midpoint splitting rule for box-decomposition trees
//
//		This is the simplest splitting rule that guarantees boxes
//		of bounded aspect ratio.  It simply cuts the box with the
//		longest side through its midpoint.  If there are ties, it
//		selects the dimension with the maximum point spread.
//
//		WARNING: This routine (while simple) doesn't seem to work
//		well in practice in high dimensions, because it tends to
//		generate a large number of trivial and/or unbalanced splits.
//		Either kd_split(), sl_midpt_split(), or fair_split() are
//		recommended, instead.
//----------------------------------------------------------------------

void midpt_split(ANNpointArray pa,        // point array
                 ANNidxArray pidx,        // point indices (permuted on return)
                 const ANNorthRect &bnds, // bounding rectangle for cell
                 int n,                   // number of points
                 int dim,                 // dimension of space
                 int &cut_dim,            // cutting dimension (returned)
                 ANNcoord &cut_val,       // cutting value (returned)
                 int &n_lo)               // num of points on low side (returned)
{
    int d;

    ANNcoord max_length = bnds.hi[0] - bnds.lo[0];
    for (d = 1; d < dim; d++) { // find length of longest box side
        ANNcoord length = bnds.hi[d] - bnds.lo[d];
        if (length > max_length) {
            max_length = length;
        }
    }
    ANNcoord max_spread = -1; // find long side with most spread
    for (d = 0; d < dim; d++) {
        // is it among longest?
        if (double(bnds.hi[d] - bnds.lo[d]) >= (1 - ERR) * max_length) {
            // compute its spread
            ANNcoord spr = annSpread(pa, pidx, n, d);
            if (spr > max_spread) { // is it max so far?
                max_spread = spr;
                cut_dim    = d;
            }
        }
    }
    // split along cut_dim at midpoint
    cut_val = (bnds.lo[cut_dim] + bnds.hi[cut_dim]) / 2;
    // permute points accordingly
    int br1, br2;
    annPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
    //------------------------------------------------------------------
    //	On return:		pa[0..br1-1] < cut_val
    //					pa[br1..br2-1] == cut_val
    //					pa[br2..n-1] > cut_val
    //
    //	We can set n_lo to any value in the range [br1..br2].
    //	We choose split so that points are most evenly divided.
    //------------------------------------------------------------------
    if (br1 > n / 2)
        n_lo = br1;
    else if (br2 < n / 2)
        n_lo = br2;
    else
        n_lo = n / 2;
}

//----------------------------------------------------------------------
//	sl_midpt_split - sliding midpoint splitting rule
//
//		This is a modification of midpt_split, which has the nonsensical
//		name "sliding midpoint".  The idea is that we try to use the
//		midpoint rule, by bisecting the longest side.  If there are
//		ties, the dimension with the maximum spread is selected.  If,
//		however, the midpoint split produces a trivial split (no points
//		on one side of the splitting plane) then we slide the splitting
//		(maintaining its orientation) until it produces a nontrivial
//		split. For example, if the splitting plane is along the x-axis,
//		and all the data points have x-coordinate less than the x-bisector,
//		then the split is taken along the maximum x-coordinate of the
//		data points.
//
//		Intuitively, this rule cannot generate trivial splits, and
//		hence avoids midpt_split's tendency to produce trees with
//		a very large number of nodes.
//
//----------------------------------------------------------------------

void sl_midpt_split(ANNpointArray pa,        // point array
                    ANNidxArray pidx,        // point indices (permuted on return)
                    const ANNorthRect &bnds, // bounding rectangle for cell
                    int n,                   // number of points
                    int dim,                 // dimension of space
                    int &cut_dim,            // cutting dimension (returned)
                    ANNcoord &cut_val,       // cutting value (returned)
                    int &n_lo)               // num of points on low side (returned)
{
    int d;

    ANNcoord max_length = bnds.hi[0] - bnds.lo[0];
    for (d = 1; d < dim; d++) { // find length of longest box side
        ANNcoord length = bnds.hi[d] - bnds.lo[d];
        if (length > max_length) {
            max_length = length;
        }
    }
    ANNcoord max_spread = -1; // find long side with most spread
    for (d = 0; d < dim; d++) {
        // is it among longest?
        if ((bnds.hi[d] - bnds.lo[d]) >= (1 - ERR) * max_length) {
            // compute its spread
            ANNcoord spr = annSpread(pa, pidx, n, d);
            if (spr > max_spread) { // is it max so far?
                max_spread = spr;
                cut_dim    = d;
            }
        }
    }
    // ideal split at midpoint
    ANNcoord ideal_cut_val = (bnds.lo[cut_dim] + bnds.hi[cut_dim]) / 2;

    ANNcoord min, max;
    annMinMax(pa, pidx, n, cut_dim, min, max); // find min/max coordinates

    if (ideal_cut_val < min) // slide to min or max as needed
        cut_val = min;
    else if (ideal_cut_val > max)
        cut_val = max;
    else
        cut_val = ideal_cut_val;

    // permute points accordingly
    int br1, br2;
    annPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
    //------------------------------------------------------------------
    //	On return:		pa[0..br1-1] < cut_val
    //					pa[br1..br2-1] == cut_val
    //					pa[br2..n-1] > cut_val
    //
    //	We can set n_lo to any value in the range [br1..br2] to satisfy
    //	the exit conditions of the procedure.
    //
    //	if ideal_cut_val < min (implying br2 >= 1),
    //			then we select n_lo = 1 (so there is one point on left) and
    //	if ideal_cut_val > max (implying br1 <= n-1),
    //			then we select n_lo = n-1 (so there is one point on right).
    //	Otherwise, we select n_lo as close to n/2 as possible within
    //			[br1..br2].
    //------------------------------------------------------------------
    if (ideal_cut_val < min)
        n_lo = 1;
    else if (ideal_cut_val > max)
        n_lo = n - 1;
    else if (br1 > n / 2)
        n_lo = br1;
    else if (br2 < n / 2)
        n_lo = br2;
    else
        n_lo = n / 2;
}

//----------------------------------------------------------------------
//	fair_split - fair-split splitting rule
//
//		This is a compromise between the kd-tree splitting rule (which
//		always splits data points at their median) and the midpoint
//		splitting rule (which always splits a box through its center.
//		The goal of this procedure is to achieve both nicely balanced
//		splits, and boxes of bounded aspect ratio.
//
//		A constant FS_ASPECT_RATIO is defined. Given a box, those sides
//		which can be split so that the ratio of the longest to shortest
//		side does not exceed ASPECT_RATIO are identified.  Among these
//		sides, we select the one in which the points have the largest
//		spread. We then split the points in a manner which most evenly
//		distributes the points on either side of the splitting plane,
//		subject to maintaining the bound on the ratio of long to short
//		sides. To determine that the aspect ratio will be preserved,
//		we determine the longest side (other than this side), and
//		determine how narrowly we can cut this side, without causing the
//		aspect ratio bound to be exceeded (small_piece).
//
//		This procedure is more robust than either kd_split or midpt_split,
//		but is more complicated as well.  When point distribution is
//		extremely skewed, this degenerates to midpt_split (actually
//		1/3 point split), and when the points are most evenly distributed,
//		this degenerates to kd-split.
//----------------------------------------------------------------------

void fair_split(ANNpointArray pa,        // point array
                ANNidxArray pidx,        // point indices (permuted on return)
                const ANNorthRect &bnds, // bounding rectangle for cell
                int n,                   // number of points
                int dim,                 // dimension of space
                int &cut_dim,            // cutting dimension (returned)
                ANNcoord &cut_val,       // cutting value (returned)
                int &n_lo)               // num of points on low side (returned)
{
    int d;
    ANNcoord max_length = bnds.hi[0] - bnds.lo[0];
    cut_dim             = 0;
    for (d = 1; d < dim; d++) { // find length of longest box side
        ANNcoord length = bnds.hi[d] - bnds.lo[d];
        if (length > max_length) {
            max_length = length;
            cut_dim    = d;
        }
    }

    ANNcoord max_spread = 0; // find legal cut with max spread
    cut_dim             = 0;
    for (d = 0; d < dim; d++) {
        ANNcoord length = bnds.hi[d] - bnds.lo[d];
        // is this side midpoint splitable
        // without violating aspect ratio?
        if (((double) max_length) * 2.0 / ((double) length) <= FS_ASPECT_RATIO) {
            // compute spread along this dim
            ANNcoord spr = annSpread(pa, pidx, n, d);
            if (spr > max_spread) { // best spread so far
                max_spread = spr;
                cut_dim    = d; // this is dimension to cut
            }
        }
    }

    max_length = 0; // find longest side other than cut_dim
    for (d = 0; d < dim; d++) {
        ANNcoord length = bnds.hi[d] - bnds.lo[d];
        if (d != cut_dim && length > max_length)
            max_length = length;
    }
    // consider most extreme splits
    ANNcoord small_piece = max_length / FS_ASPECT_RATIO;
    ANNcoord lo_cut      = bnds.lo[cut_dim] + small_piece; // lowest legal cut
    ANNcoord hi_cut      = bnds.hi[cut_dim] - small_piece; // highest legal cut

    int br1, br2;
    // is median below lo_cut ?
    if (annSplitBalance(pa, pidx, n, cut_dim, lo_cut) >= 0) {
        cut_val = lo_cut; // cut at lo_cut
        annPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
        n_lo = br1;
    }
    // is median above hi_cut?
    else if (annSplitBalance(pa, pidx, n, cut_dim, hi_cut) <= 0) {
        cut_val = hi_cut; // cut at hi_cut
        annPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
        n_lo = br2;
    } else {          // median cut preserves asp ratio
        n_lo = n / 2; // split about median
        annMedianSplit(pa, pidx, n, cut_dim, cut_val, n_lo);
    }
}

//----------------------------------------------------------------------
//	sl_fair_split - sliding fair split splitting rule
//
//		Sliding fair split is a splitting rule that combines the
//		strengths of both fair split with sliding midpoint split.
//		Fair split tends to produce balanced splits when the points
//		are roughly uniformly distributed, but it can produce many
//		trivial splits when points are highly clustered.  Sliding
//		midpoint never produces trivial splits, and shrinks boxes
//		nicely if points are highly clustered, but it may produce
//		rather unbalanced splits when points are unclustered but not
//		quite uniform.
//
//		Sliding fair split is based on the theory that there are two
//		types of splits that are "good": balanced splits that produce
//		fat boxes, and unbalanced splits provided the cell with fewer
//		points is fat.
//
//		This splitting rule operates by first computing the longest
//		side of the current bounding box.  Then it asks which sides
//		could be split (at the midpoint) and still satisfy the aspect
//		ratio bound with respect to this side.	Among these, it selects
//		the side with the largest spread (as fair split would).	 It
//		then considers the most extreme cuts that would be allowed by
//		the aspect ratio bound.	 This is done by dividing the longest
//		side of the box by the aspect ratio bound.	If the median cut
//		lies between these extreme cuts, then we use the median cut.
//		If not, then consider the extreme cut that is closer to the
//		median.	 If all the points lie to one side of this cut, then
//		we slide the cut until it hits the first point.	 This may
//		violate the aspect ratio bound, but will never generate empty
//		cells.	However the sibling of every such skinny cell is fat,
//		and hence packing arguments still apply.
//
//----------------------------------------------------------------------

void sl_fair_split(ANNpointArray pa,        // point array
                   ANNidxArray pidx,        // point indices (permuted on return)
                   const ANNorthRect &bnds, // bounding rectangle for cell
                   int n,                   // number of points
                   int dim,                 // dimension of space
                   int &cut_dim,            // cutting dimension (returned)
                   ANNcoord &cut_val,       // cutting value (returned)
                   int &n_lo)               // num of points on low side (returned)
{
    int d;
    ANNcoord min, max; // min/max coordinates
    int br1, br2;      // split break points

    ANNcoord max_length = bnds.hi[0] - bnds.lo[0];
    cut_dim             = 0;
    for (d = 1; d < dim; d++) { // find length of longest box side
        ANNcoord length = bnds.hi[d] - bnds.lo[d];
        if (length > max_length) {
            max_length = length;
            cut_dim    = d;
        }
    }

    ANNcoord max_spread = 0; // find legal cut with max spread
    cut_dim             = 0;
    for (d = 0; d < dim; d++) {
        ANNcoord length = bnds.hi[d] - bnds.lo[d];
        // is this side midpoint splitable
        // without violating aspect ratio?
        if (((double) max_length) * 2.0 / ((double) length) <= FS_ASPECT_RATIO) {
            // compute spread along this dim
            ANNcoord spr = annSpread(pa, pidx, n, d);
            if (spr > max_spread) { // best spread so far
                max_spread = spr;
                cut_dim    = d; // this is dimension to cut
            }
        }
    }

    max_length = 0; // find longest side other than cut_dim
    for (d = 0; d < dim; d++) {
        ANNcoord length = bnds.hi[d] - bnds.lo[d];
        if (d != cut_dim && length > max_length)
            max_length = length;
    }
    // consider most extreme splits
    ANNcoord small_piece = max_length / FS_ASPECT_RATIO;
    ANNcoord lo_cut      = bnds.lo[cut_dim] + small_piece; // lowest legal cut
    ANNcoord hi_cut      = bnds.hi[cut_dim] - small_piece; // highest legal cut
                                                           // find min and max along cut_dim
    annMinMax(pa, pidx, n, cut_dim, min, max);
    // is median below lo_cut?
    if (annSplitBalance(pa, pidx, n, cut_dim, lo_cut) >= 0) {
        if (max > lo_cut) {   // are any points above lo_cut?
            cut_val = lo_cut; // cut at lo_cut
            annPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
            n_lo = br1;    // balance if there are ties
        } else {           // all points below lo_cut
            cut_val = max; // cut at max value
            annPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
            n_lo = n - 1;
        }
    }
    // is median above hi_cut?
    else if (annSplitBalance(pa, pidx, n, cut_dim, hi_cut) <= 0) {
        if (min < hi_cut) {   // are any points below hi_cut?
            cut_val = hi_cut; // cut at hi_cut
            annPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
            n_lo = br2;    // balance if there are ties
        } else {           // all points above hi_cut
            cut_val = min; // cut at min value
            annPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
            n_lo = 1;
        }
    } else {          // median cut is good enough
        n_lo = n / 2; // split about median
        annMedianSplit(pa, pidx, n, cut_dim, cut_val, n_lo);
    }
}
//----------------------------------------------------------------------
// File:			kd_tree.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Basic methods for kd-trees.
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.0  04/01/05
//		Increased aspect ratio bound (ANN_AR_TOOBIG) from 100 to 1000.
//		Fixed leaf counts to count trivial leaves.
//		Added optional pa, pi arguments to Skeleton kd_tree constructor
//			for use in load constructor.
//		Added annClose() to eliminate KD_TRIVIAL memory leak.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Global data
//
//	For some splitting rules, especially with small bucket sizes,
//	it is possible to generate a large number of empty leaf nodes.
//	To save storage we allocate a single trivial leaf node which
//	contains no points.  For messy coding reasons it is convenient
//	to have it reference a trivial point index.
//
//	KD_TRIVIAL is allocated when the first kd-tree is created.  It
//	must *never* deallocated (since it may be shared by more than
//	one tree).
//----------------------------------------------------------------------
static int IDX_TRIVIAL[] = { 0 }; // trivial point index
ANNkd_leaf *KD_TRIVIAL   = NULL;  // trivial leaf node

//----------------------------------------------------------------------
//	Printing the kd-tree
//		These routines print a kd-tree in reverse inorder (high then
//		root then low).  (This is so that if you look at the output
//		from the right side it appear from left to right in standard
//		inorder.)  When outputting leaves we output only the point
//		indices rather than the point coordinates. There is an option
//		to print the point coordinates separately.
//
//		The tree printing routine calls the printing routines on the
//		individual nodes of the tree, passing in the level or depth
//		in the tree.  The level in the tree is used to print indentation
//		for readability.
//----------------------------------------------------------------------

void ANNkd_split::print( // print splitting node
    int level,           // depth of node in tree
    ostream &out)        // output stream
{
    child[ANN_HI]->print(level + 1, out); // print high child
    out << "    ";
    for (int i = 0; i < level; i++) // print indentation
        out << "..";
    out << "Split cd=" << cut_dim << " cv=" << cut_val;
    out << " lbnd=" << cd_bnds[ANN_LO];
    out << " hbnd=" << cd_bnds[ANN_HI];
    out << "\n";
    child[ANN_LO]->print(level + 1, out); // print low child
}

void ANNkd_leaf::print( // print leaf node
    int level,          // depth of node in tree
    ostream &out)       // output stream
{

    out << "    ";
    for (int i = 0; i < level; i++) // print indentation
        out << "..";

    if (this == KD_TRIVIAL) { // canonical trivial leaf node
        out << "Leaf (trivial)\n";
    } else {
        out << "Leaf n=" << n_pts << " <";
        for (int j = 0; j < n_pts; j++) {
            out << bkt[j];
            if (j < n_pts - 1)
                out << ",";
        }
        out << ">\n";
    }
}

void ANNkd_tree::Print( // print entire tree
    ANNbool with_pts,   // print points as well?
    ostream &out)       // output stream
{
    out << "ANN Version " << ANNversion << "\n";
    if (with_pts) { // print point coordinates
        out << "    Points:\n";
        for (int i = 0; i < n_pts; i++) {
            out << "\t" << i << ": ";
            annPrintPt(pts[i], dim, out);
            out << "\n";
        }
    }
    if (root == NULL) // empty tree?
        out << "    Null tree.\n";
    else {
        root->print(0, out); // invoke printing at root
    }
}

//----------------------------------------------------------------------
//	kd_tree statistics (for performance evaluation)
//		This routine compute various statistics information for
//		a kd-tree.  It is used by the implementors for performance
//		evaluation of the data structure.
//----------------------------------------------------------------------

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void ANNkdStats::merge(const ANNkdStats &st) // merge stats from child
{
    n_lf += st.n_lf;
    n_tl += st.n_tl;
    n_spl += st.n_spl;
    n_shr += st.n_shr;
    depth = MAX(depth, st.depth);
    sum_ar += st.sum_ar;
}

//----------------------------------------------------------------------
//	Update statistics for nodes
//----------------------------------------------------------------------

const double ANN_AR_TOOBIG = 1000; // too big an aspect ratio

void ANNkd_leaf::getStats( // get subtree statistics
    int dim,               // dimension of space
    ANNkdStats &st,        // stats (modified)
    ANNorthRect &bnd_box)  // bounding box
{
    st.reset();
    st.n_lf = 1; // count this leaf
    if (this == KD_TRIVIAL)
        st.n_tl = 1;                          // count trivial leaf
    double ar = annAspectRatio(dim, bnd_box); // aspect ratio of leaf
                                              // incr sum (ignore outliers)
    st.sum_ar += float(ar < ANN_AR_TOOBIG ? ar : ANN_AR_TOOBIG);
}

void ANNkd_split::getStats( // get subtree statistics
    int dim,                // dimension of space
    ANNkdStats &st,         // stats (modified)
    ANNorthRect &bnd_box)   // bounding box
{
    ANNkdStats ch_stats;                       // stats for children
                                               // get stats for low child
    ANNcoord hv         = bnd_box.hi[cut_dim]; // save box bounds
    bnd_box.hi[cut_dim] = cut_val;             // upper bound for low child
    ch_stats.reset();                          // reset
    child[ANN_LO]->getStats(dim, ch_stats, bnd_box);
    st.merge(ch_stats);                        // merge them
    bnd_box.hi[cut_dim] = hv;                  // restore bound
                                               // get stats for high child
    ANNcoord lv         = bnd_box.lo[cut_dim]; // save box bounds
    bnd_box.lo[cut_dim] = cut_val;             // lower bound for high child
    ch_stats.reset();                          // reset
    child[ANN_HI]->getStats(dim, ch_stats, bnd_box);
    st.merge(ch_stats);       // merge them
    bnd_box.lo[cut_dim] = lv; // restore bound

    st.depth++; // increment depth
    st.n_spl++; // increment number of splits
}

//----------------------------------------------------------------------
//	getStats
//		Collects a number of statistics related to kd_tree or
//		bd_tree.
//----------------------------------------------------------------------

void ANNkd_tree::getStats( // get tree statistics
    ANNkdStats &st)        // stats (modified)
{
    st.reset(dim, n_pts, bkt_size); // reset stats
                                    // create bounding box
    ANNorthRect bnd_box(dim, bnd_box_lo, bnd_box_hi);
    if (root != NULL) {                   // if nonempty tree
        root->getStats(dim, st, bnd_box); // get statistics
        st.avg_ar = st.sum_ar / st.n_lf;  // average leaf asp ratio
    }
}

//----------------------------------------------------------------------
//	kd_tree destructor
//		The destructor just frees the various elements that were
//		allocated in the construction process.
//----------------------------------------------------------------------

ANNkd_tree::~ANNkd_tree() // tree destructor
{
    if (root != NULL)
        delete root;
    if (pidx != NULL)
        delete[] pidx;
    if (bnd_box_lo != NULL)
        annDeallocPt(bnd_box_lo);
    if (bnd_box_hi != NULL)
        annDeallocPt(bnd_box_hi);
}

//----------------------------------------------------------------------
//	This is called with all use of ANN is finished.  It eliminates the
//	minor memory leak caused by the allocation of KD_TRIVIAL.
//----------------------------------------------------------------------
void annClose() // close use of ANN
{
    if (KD_TRIVIAL != NULL) {
        delete KD_TRIVIAL;
        KD_TRIVIAL = NULL;
    }
}

//----------------------------------------------------------------------
//	kd_tree constructors
//		There is a skeleton kd-tree constructor which sets up a
//		trivial empty tree.	 The last optional argument allows
//		the routine to be passed a point index array which is
//		assumed to be of the proper size (n).  Otherwise, one is
//		allocated and initialized to the identity.	Warning: In
//		either case the destructor will deallocate this array.
//
//		As a kludge, we need to allocate KD_TRIVIAL if one has not
//		already been allocated.	 (This is because I'm too dumb to
//		figure out how to cause a pointer to be allocated at load
//		time.)
//----------------------------------------------------------------------

void ANNkd_tree::SkeletonTree( // construct skeleton tree
    int n,                     // number of points
    int dd,                    // dimension
    int bs,                    // bucket size
    ANNpointArray pa,          // point array
    ANNidxArray pi)            // point indices
{
    dim      = dd; // initialize basic elements
    n_pts    = n;
    bkt_size = bs;
    pts      = pa; // initialize points array

    root = NULL; // no associated tree yet

    if (pi == NULL) {         // point indices provided?
        pidx = new ANNidx[n]; // no, allocate space for point indices
        for (int i = 0; i < n; i++) {
            pidx[i] = i; // initially identity
        }
    } else {
        pidx = pi; // yes, use them
    }

    bnd_box_lo = bnd_box_hi = NULL;                  // bounding box is nonexistent
    if (KD_TRIVIAL == NULL)                          // no trivial leaf node yet?
        KD_TRIVIAL = new ANNkd_leaf(0, IDX_TRIVIAL); // allocate it
}

ANNkd_tree::ANNkd_tree( // basic constructor
    int n,              // number of points
    int dd,             // dimension
    int bs)             // bucket size
{
    SkeletonTree(n, dd, bs);
} // construct skeleton tree

//----------------------------------------------------------------------
//	rkd_tree - recursive procedure to build a kd-tree
//
//		Builds a kd-tree for points in pa as indexed through the
//		array pidx[0..n-1] (typically a subarray of the array used in
//		the top-level call).  This routine permutes the array pidx,
//		but does not alter pa[].
//
//		The construction is based on a standard algorithm for constructing
//		the kd-tree (see Friedman, Bentley, and Finkel, ``An algorithm for
//		finding best matches in logarithmic expected time,'' ACM Transactions
//		on Mathematical Software, 3(3):209-226, 1977).  The procedure
//		operates by a simple divide-and-conquer strategy, which determines
//		an appropriate orthogonal cutting plane (see below), and splits
//		the points.  When the number of points falls below the bucket size,
//		we simply store the points in a leaf node's bucket.
//
//		One of the arguments is a pointer to a splitting routine,
//		whose prototype is:
//
//				void split(
//						ANNpointArray pa,  // complete point array
//						ANNidxArray pidx,  // point array (permuted on return)
//						ANNorthRect &bnds, // bounds of current cell
//						int n,			   // number of points
//						int dim,		   // dimension of space
//						int &cut_dim,	   // cutting dimension
//						ANNcoord &cut_val, // cutting value
//						int &n_lo)		   // no. of points on low side of cut
//
//		This procedure selects a cutting dimension and cutting value,
//		partitions pa about these values, and returns the number of
//		points on the low side of the cut.
//----------------------------------------------------------------------

ANNkd_ptr rkd_tree(          // recursive construction of kd-tree
    ANNpointArray pa,        // point array
    ANNidxArray pidx,        // point indices to store in subtree
    int n,                   // number of points
    int dim,                 // dimension of space
    int bsp,                 // bucket space
    ANNorthRect &bnd_box,    // bounding box for current node
    ANNkd_splitter splitter) // splitting routine
{
    if (n <= bsp) {            // n small, make a leaf node
        if (n == 0)            // empty leaf node
            return KD_TRIVIAL; // return (canonical) empty leaf
        else                   // construct the node and return
            return new ANNkd_leaf(n, pidx);
    } else {                 // n large, make a splitting node
        int cd;              // cutting dimension
        ANNcoord cv;         // cutting value
        int n_lo;            // number on low side of cut
        ANNkd_node *lo, *hi; // low and high children

        // invoke splitting procedure
        (*splitter)(pa, pidx, bnd_box, n, dim, cd, cv, n_lo);

        ANNcoord lv = bnd_box.lo[cd]; // save bounds for cutting dimension
        ANNcoord hv = bnd_box.hi[cd];

        bnd_box.hi[cd] = cv;            // modify bounds for left subtree
        lo             = rkd_tree(      // build left subtree
            pa, pidx, n_lo, // ...from pidx[0..n_lo-1]
            dim, bsp, bnd_box, splitter);
        bnd_box.hi[cd] = hv; // restore bounds

        bnd_box.lo[cd] = cv;                       // modify bounds for right subtree
        hi             = rkd_tree(                 // build right subtree
            pa, pidx + n_lo, n - n_lo, // ...from pidx[n_lo..n-1]
            dim, bsp, bnd_box, splitter);
        bnd_box.lo[cd] = lv; // restore bounds

        // create the splitting node
        ANNkd_split *ptr = new ANNkd_split(cd, cv, lv, hv, lo, hi);

        return ptr; // return pointer to this node
    }
}

//----------------------------------------------------------------------
// kd-tree constructor
//		This is the main constructor for kd-trees given a set of points.
//		It first builds a skeleton tree, then computes the bounding box
//		of the data points, and then invokes rkd_tree() to actually
//		build the tree, passing it the appropriate splitting routine.
//----------------------------------------------------------------------

ANNkd_tree::ANNkd_tree( // construct from point array
    ANNpointArray pa,   // point array (with at least n pts)
    int n,              // number of points
    int dd,             // dimension
    int bs,             // bucket size
    ANNsplitRule split) // splitting method
{
    SkeletonTree(n, dd, bs); // set up the basic stuff
    pts = pa;                // where the points are
    if (n == 0)
        return; // no points--no sweat

    ANNorthRect bnd_box(dd);               // bounding box for points
    annEnclRect(pa, pidx, n, dd, bnd_box); // construct bounding rectangle
                                           // copy to tree structure
    bnd_box_lo = annCopyPt(dd, bnd_box.lo);
    bnd_box_hi = annCopyPt(dd, bnd_box.hi);

    switch (split) {     // build by rule
        case ANN_KD_STD: // standard kd-splitting rule
            root = rkd_tree(pa, pidx, n, dd, bs, bnd_box, kd_split);
            break;
        case ANN_KD_MIDPT: // midpoint split
            root = rkd_tree(pa, pidx, n, dd, bs, bnd_box, midpt_split);
            break;
        case ANN_KD_FAIR: // fair split
            root = rkd_tree(pa, pidx, n, dd, bs, bnd_box, fair_split);
            break;
        case ANN_KD_SUGGEST:  // best (in our opinion)
        case ANN_KD_SL_MIDPT: // sliding midpoint split
            root = rkd_tree(pa, pidx, n, dd, bs, bnd_box, sl_midpt_split);
            break;
        case ANN_KD_SL_FAIR: // sliding fair split
            root = rkd_tree(pa, pidx, n, dd, bs, bnd_box, sl_fair_split);
            break;
        default:
            annError("Illegal splitting method", ANNabort);
    }
}
//----------------------------------------------------------------------
// File:			kd_util.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Common utilities for kd-trees
// Last modified:	01/04/05 (Version 1.0)
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// The following routines are utility functions for manipulating
// points sets, used in determining splitting planes for kd-tree
// construction.
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	NOTE: Virtually all point indexing is done through an index (i.e.
//	permutation) array pidx.  Consequently, a reference to the d-th
//	coordinate of the i-th point is pa[pidx[i]][d].  The macro PA(i,d)
//	is a shorthand for this.
//----------------------------------------------------------------------
// standard 2-d indirect indexing
#define PA(i, d) (pa[pidx[(i)]][(d)])
// accessing a single point
#define PP(i) (pa[pidx[(i)]])

//----------------------------------------------------------------------
//	annAspectRatio
//		Compute the aspect ratio (ratio of longest to shortest side)
//		of a rectangle.
//----------------------------------------------------------------------

double annAspectRatio(int dim,                    // dimension
                      const ANNorthRect &bnd_box) // bounding cube
{
    ANNcoord length     = bnd_box.hi[0] - bnd_box.lo[0];
    ANNcoord min_length = length; // min side length
    ANNcoord max_length = length; // max side length
    for (int d = 0; d < dim; d++) {
        length = bnd_box.hi[d] - bnd_box.lo[d];
        if (length < min_length)
            min_length = length;
        if (length > max_length)
            max_length = length;
    }
    return max_length / min_length;
}

//----------------------------------------------------------------------
//	annEnclRect, annEnclCube
//		These utilities compute the smallest rectangle and cube enclosing
//		a set of points, respectively.
//----------------------------------------------------------------------

void annEnclRect(ANNpointArray pa,  // point array
                 ANNidxArray pidx,  // point indices
                 int n,             // number of points
                 int dim,           // dimension
                 ANNorthRect &bnds) // bounding cube (returned)
{
    for (int d = 0; d < dim; d++) { // find smallest enclosing rectangle
        ANNcoord lo_bnd = PA(0, d); // lower bound on dimension d
        ANNcoord hi_bnd = PA(0, d); // upper bound on dimension d
        for (int i = 0; i < n; i++) {
            if (PA(i, d) < lo_bnd)
                lo_bnd = PA(i, d);
            else if (PA(i, d) > hi_bnd)
                hi_bnd = PA(i, d);
        }
        bnds.lo[d] = lo_bnd;
        bnds.hi[d] = hi_bnd;
    }
}

void annEnclCube(      // compute smallest enclosing cube
    ANNpointArray pa,  // point array
    ANNidxArray pidx,  // point indices
    int n,             // number of points
    int dim,           // dimension
    ANNorthRect &bnds) // bounding cube (returned)
{
    int d;
    // compute smallest enclosing rect
    annEnclRect(pa, pidx, n, dim, bnds);

    ANNcoord max_len = 0;       // max length of any side
    for (d = 0; d < dim; d++) { // determine max side length
        ANNcoord len = bnds.hi[d] - bnds.lo[d];
        if (len > max_len) { // update max_len if longest
            max_len = len;
        }
    }
    for (d = 0; d < dim; d++) { // grow sides to match max
        ANNcoord len       = bnds.hi[d] - bnds.lo[d];
        ANNcoord half_diff = (max_len - len) / 2;
        bnds.lo[d] -= half_diff;
        bnds.hi[d] += half_diff;
    }
}

//----------------------------------------------------------------------
//	annBoxDistance - utility routine which computes distance from point to
//		box (Note: most distances to boxes are computed using incremental
//		distance updates, not this function.)
//----------------------------------------------------------------------

ANNdist annBoxDistance( // compute distance from point to box
    const ANNpoint q,   // the point
    const ANNpoint lo,  // low point of box
    const ANNpoint hi,  // high point of box
    int dim)            // dimension of space
{
    register ANNdist dist = 0.0; // sum of squared distances
    register ANNdist t;

    for (register int d = 0; d < dim; d++) {
        if (q[d] < lo[d]) { // q is left of box
            t    = ANNdist(lo[d]) - ANNdist(q[d]);
            dist = ANN_SUM(dist, ANN_POW(t));
        } else if (q[d] > hi[d]) { // q is right of box
            t    = ANNdist(q[d]) - ANNdist(hi[d]);
            dist = ANN_SUM(dist, ANN_POW(t));
        }
    }
    ANN_FLOP(4 * dim) // increment floating op count

    return dist;
}

//----------------------------------------------------------------------
//	annSpread - find spread along given dimension
//	annMinMax - find min and max coordinates along given dimension
//	annMaxSpread - find dimension of max spread
//----------------------------------------------------------------------

ANNcoord annSpread(   // compute point spread along dimension
    ANNpointArray pa, // point array
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d)            // dimension to check
{
    ANNcoord min = PA(0, d); // compute max and min coords
    ANNcoord max = PA(0, d);
    for (int i = 1; i < n; i++) {
        ANNcoord c = PA(i, d);
        if (c < min)
            min = c;
        else if (c > max)
            max = c;
    }
    return (max - min); // total spread is difference
}

void annMinMax(       // compute min and max coordinates along dim
    ANNpointArray pa, // point array
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d,            // dimension to check
    ANNcoord &min,    // minimum value (returned)
    ANNcoord &max)    // maximum value (returned)
{
    min = PA(0, d); // compute max and min coords
    max = PA(0, d);
    for (int i = 1; i < n; i++) {
        ANNcoord c = PA(i, d);
        if (c < min)
            min = c;
        else if (c > max)
            max = c;
    }
}

int annMaxSpread(     // compute dimension of max spread
    ANNpointArray pa, // point array
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int dim)          // dimension of space
{
    int max_dim      = 0; // dimension of max spread
    ANNcoord max_spr = 0; // amount of max spread

    if (n == 0)
        return max_dim; // no points, who cares?

    for (int d = 0; d < dim; d++) { // compute spread along each dim
        ANNcoord spr = annSpread(pa, pidx, n, d);
        if (spr > max_spr) { // bigger than current max
            max_spr = spr;
            max_dim = d;
        }
    }
    return max_dim;
}

//----------------------------------------------------------------------
//	annMedianSplit - split point array about its median
//		Splits a subarray of points pa[0..n] about an element of given
//		rank (median: n_lo = n/2) with respect to dimension d.  It places
//		the element of rank n_lo-1 correctly (because our splitting rule
//		takes the mean of these two).  On exit, the array is permuted so
//		that:
//
//		pa[0..n_lo-2][d] <= pa[n_lo-1][d] <= pa[n_lo][d] <= pa[n_lo+1..n-1][d].
//
//		The mean of pa[n_lo-1][d] and pa[n_lo][d] is returned as the
//		splitting value.
//
//		All indexing is done indirectly through the index array pidx.
//
//		This function uses the well known selection algorithm due to
//		C.A.R. Hoare.
//----------------------------------------------------------------------

// swap two points in pa array
#define PASWAP(a, b)                                                                                                   \
    {                                                                                                                  \
        int tmp = pidx[a];                                                                                             \
        pidx[a] = pidx[b];                                                                                             \
        pidx[b] = tmp;                                                                                                 \
    }

void annMedianSplit(ANNpointArray pa, // points to split
                    ANNidxArray pidx, // point indices
                    int n,            // number of points
                    int d,            // dimension along which to split
                    ANNcoord &cv,     // cutting value
                    int n_lo)         // split into n_lo and n-n_lo
{
    int l = 0;     // left end of current subarray
    int r = n - 1; // right end of current subarray
    while (l < r) {
        register int i = (r + l) / 2; // select middle as pivot
        register int k;

        if (PA(i, d) > PA(r, d)) // make sure last > pivot
            PASWAP(i, r)
        PASWAP(l, i); // move pivot to first position

        ANNcoord c = PA(l, d); // pivot value
        i          = l;
        k          = r;
        for (;;) { // pivot about c
            while (PA(++i, d) < c)
                ;
            while (PA(--k, d) > c)
                ;
            if (i < k)
                PASWAP(i, k) else break;
        }
        PASWAP(l, k); // pivot winds up in location k

        if (k > n_lo)
            r = k - 1; // recurse on proper subarray
        else if (k < n_lo)
            l = k + 1;
        else
            break; // got the median exactly
    }
    if (n_lo > 0) {            // search for next smaller item
        ANNcoord c = PA(0, d); // candidate for max
        int k      = 0;        // candidate's index
        for (int i = 1; i < n_lo; i++) {
            if (PA(i, d) > c) {
                c = PA(i, d);
                k = i;
            }
        }
        PASWAP(n_lo - 1, k); // max among pa[0..n_lo-1] to pa[n_lo-1]
    }
    // cut value is midpoint value
    cv = (PA(n_lo - 1, d) + PA(n_lo, d)) / 2.0;
}

//----------------------------------------------------------------------
//	annPlaneSplit - split point array about a cutting plane
//		Split the points in an array about a given plane along a
//		given cutting dimension.  On exit, br1 and br2 are set so
//		that:
//
//				pa[ 0 ..br1-1] <  cv
//				pa[br1..br2-1] == cv
//				pa[br2.. n -1] >  cv
//
//		All indexing is done indirectly through the index array pidx.
//
//----------------------------------------------------------------------

void annPlaneSplit(   // split points by a plane
    ANNpointArray pa, // points to split
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d,            // dimension along which to split
    ANNcoord cv,      // cutting value
    int &br1,         // first break (values < cv)
    int &br2)         // second break (values == cv)
{
    int l = 0;
    int r = n - 1;
    for (;;) { // partition pa[0..n-1] about cv
        while (l < n && PA(l, d) < cv)
            l++;
        while (r >= 0 && PA(r, d) >= cv)
            r--;
        if (l > r)
            break;
        PASWAP(l, r);
        l++;
        r--;
    }
    br1 = l; // now: pa[0..br1-1] < cv <= pa[br1..n-1]
    r   = n - 1;
    for (;;) { // partition pa[br1..n-1] about cv
        while (l < n && PA(l, d) <= cv)
            l++;
        while (r >= br1 && PA(r, d) > cv)
            r--;
        if (l > r)
            break;
        PASWAP(l, r);
        l++;
        r--;
    }
    br2 = l; // now: pa[br1..br2-1] == cv < pa[br2..n-1]
}

//----------------------------------------------------------------------
//	annBoxSplit - split point array about a orthogonal rectangle
//		Split the points in an array about a given orthogonal
//		rectangle.  On exit, n_in is set to the number of points
//		that are inside (or on the boundary of) the rectangle.
//
//		All indexing is done indirectly through the index array pidx.
//
//----------------------------------------------------------------------

void annBoxSplit(     // split points by a box
    ANNpointArray pa, // points to split
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int dim,          // dimension of space
    ANNorthRect &box, // the box
    int &n_in)        // number of points inside (returned)
{
    int l = 0;
    int r = n - 1;
    for (;;) { // partition pa[0..n-1] about box
        while (l < n && box.inside(dim, PP(l)))
            l++;
        while (r >= 0 && !box.inside(dim, PP(r)))
            r--;
        if (l > r)
            break;
        PASWAP(l, r);
        l++;
        r--;
    }
    n_in = l; // now: pa[0..n_in-1] inside and rest outside
}

//----------------------------------------------------------------------
//	annSplitBalance - compute balance factor for a given plane split
//		Balance factor is defined as the number of points lying
//		below the splitting value minus n/2 (median).  Thus, a
//		median split has balance 0, left of this is negative and
//		right of this is positive.  (The points are unchanged.)
//----------------------------------------------------------------------

int annSplitBalance(  // determine balance factor of a split
    ANNpointArray pa, // points to split
    ANNidxArray pidx, // point indices
    int n,            // number of points
    int d,            // dimension along which to split
    ANNcoord cv)      // cutting value
{
    int n_lo = 0;
    for (int i = 0; i < n; i++) { // count number less than cv
        if (PA(i, d) < cv)
            n_lo++;
    }
    return n_lo - n / 2;
}

//----------------------------------------------------------------------
//	annBox2Bnds - convert bounding box to list of bounds
//		Given two boxes, an inner box enclosed within a bounding
//		box, this routine determines all the sides for which the
//		inner box is strictly contained with the bounding box,
//		and adds an appropriate entry to a list of bounds.  Then
//		we allocate storage for the final list of bounds, and return
//		the resulting list and its size.
//----------------------------------------------------------------------

void annBox2Bnds(                 // convert inner box to bounds
    const ANNorthRect &inner_box, // inner box
    const ANNorthRect &bnd_box,   // enclosing box
    int dim,                      // dimension of space
    int &n_bnds,                  // number of bounds (returned)
    ANNorthHSArray &bnds)         // bounds array (returned)
{
    int i;
    n_bnds = 0; // count number of bounds
    for (i = 0; i < dim; i++) {
        if (inner_box.lo[i] > bnd_box.lo[i]) // low bound is inside
            n_bnds++;
        if (inner_box.hi[i] < bnd_box.hi[i]) // high bound is inside
            n_bnds++;
    }

    bnds = new ANNorthHalfSpace[n_bnds]; // allocate appropriate size

    int j = 0;
    for (i = 0; i < dim; i++) { // fill the array
        if (inner_box.lo[i] > bnd_box.lo[i]) {
            bnds[j].cd = i;
            bnds[j].cv = inner_box.lo[i];
            bnds[j].sd = +1;
            j++;
        }
        if (inner_box.hi[i] < bnd_box.hi[i]) {
            bnds[j].cd = i;
            bnds[j].cv = inner_box.hi[i];
            bnds[j].sd = -1;
            j++;
        }
    }
}

//----------------------------------------------------------------------
//	annBnds2Box - convert list of bounds to bounding box
//		Given an enclosing box and a list of bounds, this routine
//		computes the corresponding inner box.  It is assumed that
//		the box points have been allocated already.
//----------------------------------------------------------------------

void annBnds2Box(const ANNorthRect &bnd_box, // enclosing box
                 int dim,                    // dimension of space
                 int n_bnds,                 // number of bounds
                 ANNorthHSArray bnds,        // bounds array
                 ANNorthRect &inner_box)     // inner box (returned)
{
    annAssignRect(dim, inner_box, bnd_box); // copy bounding box to inner

    for (int i = 0; i < n_bnds; i++) {
        bnds[i].project(inner_box.lo); // project each endpoint
        bnds[i].project(inner_box.hi);
    }
}
//----------------------------------------------------------------------
// File:			perf.cpp
// Programmer:		Sunil Arya and David Mount
// Description:		Methods for performance stats
// Last modified:	01/27/10 (Version 1.1.2)
//----------------------------------------------------------------------
// Copyright (c) 1997-2010 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------
// History:
//	Revision 0.1  03/04/98
//		Initial release
//	Revision 1.0  04/01/05
//		Changed names to avoid namespace conflicts.
//		Added flush after printing performance stats to fix bug
//			in Microsoft Windows version.
//	Revision 1.1.2  01/27/10
//		Fixed minor compilation bugs for new versions of gcc
//----------------------------------------------------------------------

// make std:: available

//----------------------------------------------------------------------
//	Performance statistics
//		The following data and routines are used for computing
//		performance statistics for nearest neighbor searching.
//		Because these routines can slow the code down, they can be
//		activated and deactiviated by defining the PERF variable,
//		by compiling with the option: -DPERF
//----------------------------------------------------------------------

//----------------------------------------------------------------------
//	Global counters for performance measurement
//----------------------------------------------------------------------

int ann_Ndata_pts  = 0;    // number of data points
int ann_Nvisit_lfs = 0;    // number of leaf nodes visited
int ann_Nvisit_spl = 0;    // number of splitting nodes visited
int ann_Nvisit_shr = 0;    // number of shrinking nodes visited
int ann_Nvisit_pts = 0;    // visited points for one query
int ann_Ncoord_hts = 0;    // coordinate hits for one query
int ann_Nfloat_ops = 0;    // floating ops for one query
ANNsampStat ann_visit_lfs; // stats on leaf nodes visits
ANNsampStat ann_visit_spl; // stats on splitting nodes visits
ANNsampStat ann_visit_shr; // stats on shrinking nodes visits
ANNsampStat ann_visit_nds; // stats on total nodes visits
ANNsampStat ann_visit_pts; // stats on points visited
ANNsampStat ann_coord_hts; // stats on coordinate hits
ANNsampStat ann_float_ops; // stats on floating ops
//
ANNsampStat ann_average_err; // average error
ANNsampStat ann_rank_err;    // rank error

//----------------------------------------------------------------------
//	Routines for statistics.
//----------------------------------------------------------------------

DLL_API void annResetStats(int data_size) // reset stats for a set of queries
{
    ann_Ndata_pts = data_size;
    ann_visit_lfs.reset();
    ann_visit_spl.reset();
    ann_visit_shr.reset();
    ann_visit_nds.reset();
    ann_visit_pts.reset();
    ann_coord_hts.reset();
    ann_float_ops.reset();
    ann_average_err.reset();
    ann_rank_err.reset();
}

DLL_API void annResetCounts() // reset counts for one query
{
    ann_Nvisit_lfs = 0;
    ann_Nvisit_spl = 0;
    ann_Nvisit_shr = 0;
    ann_Nvisit_pts = 0;
    ann_Ncoord_hts = 0;
    ann_Nfloat_ops = 0;
}

DLL_API void annUpdateStats() // update stats with current counts
{
    ann_visit_lfs += ann_Nvisit_lfs;
    ann_visit_nds += ann_Nvisit_spl + ann_Nvisit_lfs;
    ann_visit_spl += ann_Nvisit_spl;
    ann_visit_shr += ann_Nvisit_shr;
    ann_visit_pts += ann_Nvisit_pts;
    ann_coord_hts += ann_Ncoord_hts;
    ann_float_ops += ann_Nfloat_ops;
}

// print a single statistic
void print_one_stat(const char *title, ANNsampStat s, double div) {
    cout << title << "= [ ";
    cout.width(9);
    cout << s.mean() / div << " : ";
    cout.width(9);
    cout << s.stdDev() / div << " ]<";
    cout.width(9);
    cout << s.min() / div << " , ";
    cout.width(9);
    cout << s.max() / div << " >\n";
}

DLL_API void annPrintStats( // print statistics for a run
    ANNbool validate)       // true if average errors desired
{
    cout.precision(4); // set floating precision
    cout << "  (Performance stats: "
         << " [      mean :    stddev ]<      min ,       max >\n";
    print_one_stat("    leaf_nodes       ", ann_visit_lfs, 1);
    print_one_stat("    splitting_nodes  ", ann_visit_spl, 1);
    print_one_stat("    shrinking_nodes  ", ann_visit_shr, 1);
    print_one_stat("    total_nodes      ", ann_visit_nds, 1);
    print_one_stat("    points_visited   ", ann_visit_pts, 1);
    print_one_stat("    coord_hits/pt    ", ann_coord_hts, ann_Ndata_pts);
    print_one_stat("    floating_ops_(K) ", ann_float_ops, 1000);
    if (validate) {
        print_one_stat("    average_error    ", ann_average_err, 1);
        print_one_stat("    rank_error       ", ann_rank_err, 1);
    }
    cout.precision(0); // restore the default
    cout << "  )\n";
    cout.flush();
}

NAMESPACE_END(mitsuba)