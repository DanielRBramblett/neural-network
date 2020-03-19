//Copyright(C) 2020 "Daniel Bramblett" <daniel.r.bramblett@gmail.com>
#include "stdafx.h"
#include "testHelperFunctions.h"
#include<math.h>

bool floatInBounds(float tested, float target, float range)
{
	return fabs(tested - target) <= range;
}