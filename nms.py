import torch
from iou import intersection_over_union

def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list #타입 리스트인지

    bboxes = [box for box in bboxes if box[1] > threshold] # prob threshold 보다 큰 점수인 박스만, confidence score 낮다면 이 과정에서 삭제한다.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # 내림차순으로 위 점수 선별
    #print(bboxes)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0) # 위에서 정렬해준 것 중 가장 윗 값(가장 큰 값)을 가져온다 (비교)
        #print("chosen box conf score", chosen_box)

        bboxes = [
            box
            for box in bboxes
            # 예측한 클래스가 다르거나
            if box[0] != chosen_box[0]
            #선택된 박스와의 iou가 threshold보다 낮다면 append -> threshold 보다 높은 값 제거
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), #x1 x2 y1 y2 만 남기기, IoU 계산
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        #선택된 박스 append
        bboxes_after_nms.append(chosen_box)

    #print("result",bboxes_after_nms)

    return bboxes_after_nms