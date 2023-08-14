#pragma once
#include <MeshFrame/Utility/Parser.h>
#include "../Types/Types.h"

namespace SP {
	enum MaterialType
	{
		NeoHookean,
		MassSpring
	};

	struct ObjectParams : public MF::BaseJsonConfig {

		ObjectParams() {
			translationBeforeScaling << 0, 0, 0;
			scale << 1, 1, 1;
			translation << 0, 0, 0;
			rotation << 0, 0, 0;
			initialVelocity << 0, 0, 0;
		}
		std::string path;

		MaterialType materialType;
		std::string materialName;

		Vec3 translationBeforeScaling;
		Vec3 scale;
		Vec3 translation;
		Vec3 rotation;
		Vec3 initialVelocity;

		FloatingType density = 1;

		bool hasNoGravZone = false;
		FloatingType noGravZoneThreshold = 0;
		FloatingType maxVelocityMagnitude = -1;
		bool shuffleParallelizationGroup = false;

		std::string tetsColoringCategoriesPath;

		typedef std::shared_ptr<ObjectParams> SharedPtr;
		
		std::vector<IdType> fixedPoints;

		virtual bool fromJson(nlohmann::json& objectParam);
		virtual bool toJson(nlohmann::json& objectParam);
	};
	
	inline bool ObjectParams::fromJson(nlohmann::json& objectParam) {
		PARSE_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, translationBeforeScaling);
		PARSE_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, scale);
		PARSE_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, translation);
		PARSE_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, rotation);
		PARSE_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, initialVelocity);

		EXTRACT_FROM_JSON(objectParam, fixedPoints);
		EXTRACT_FROM_JSON(objectParam, materialName);
		EXTRACT_FROM_JSON(objectParam, density);
		EXTRACT_FROM_JSON(objectParam, hasNoGravZone);
		EXTRACT_FROM_JSON(objectParam, noGravZoneThreshold);
		EXTRACT_FROM_JSON(objectParam, maxVelocityMagnitude);
		EXTRACT_FROM_JSON(objectParam, path);
		EXTRACT_FROM_JSON(objectParam, tetsColoringCategoriesPath);
		EXTRACT_FROM_JSON(objectParam, shuffleParallelizationGroup);
		return true;
	}

	inline bool ObjectParams::toJson(nlohmann::json& objectParam) {
		PUT_TO_JSON_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, translationBeforeScaling);
		PUT_TO_JSON_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, scale);
		PUT_TO_JSON_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, translation);
		PUT_TO_JSON_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, rotation);
		PUT_TO_JSON_VEC3_INDEXED_BY_ROUND_BRACKET(objectParam, initialVelocity);

		PUT_TO_JSON(objectParam, fixedPoints);
		PUT_TO_JSON(objectParam, materialName);
		PUT_TO_JSON(objectParam, density);
		PUT_TO_JSON(objectParam, hasNoGravZone);
		PUT_TO_JSON(objectParam, noGravZoneThreshold);
		PUT_TO_JSON(objectParam, maxVelocityMagnitude);
		PUT_TO_JSON(objectParam, path);
		PUT_TO_JSON(objectParam, tetsColoringCategoriesPath);
		PUT_TO_JSON(objectParam, shuffleParallelizationGroup);

		return true;
	}

}