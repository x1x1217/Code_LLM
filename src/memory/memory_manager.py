import json, os

class MemoryManager:
    
    def __init__(self, memory_path):
        self.memory_path = memory_path
        
        if os.path.exists(self.memory_path):
            # self.load()
            self.state = {}
            self.save()
        else:
            self.state = {}
            self.save()

    def load(self):
        with open(self.memory_path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self.state = {int(k): v for k, v in self.state.items()}
        
    def save(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.state.items()}, f, ensure_ascii=False, indent=2)
        
    def add_question(self, task_id, current_step):
        if task_id not in self.state:
            self.state[task_id] = {
                "ef": 2.5,
                "interval": 1,
                "streak": 0,
                "last_step": current_step,
                "next_step": current_step + 1,
                "graduated": False,
                "graduation_step": None,
                "skipped": False,
                "repeat_fail": 0
            }
            
    def calculate_ef(self, current_ef, quality):
        return current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            
    def check_graduation(self, task_id, current_step):
        '''graduation condition: streak >= 2'''
        
        slot = self.state[task_id]
        if not slot["graduated"]:
            if slot["streak"] >= 3:
                slot["graduated"] = True
                slot["graduation_step"] = current_step
                print(f"Task {task_id} graduated at step {current_step}.")
                return True
        return False
    
    def check_skip(self, task_id, max_repeats=5):
        slot = self.state[task_id]
        if not slot["skipped"]:
            if slot["repeat_fail"] >= max_repeats:
                slot["skipped"] = True
                print(f"Task {task_id} skipped due to repeated failures.")
                return True
        return False
    
    def update(self, task_id, pass_rate, current_step):
        if task_id not in self.state:
            self.add_question(task_id, current_step)
        
        slot = self.state[task_id]
                
        if slot["graduated"] or slot["skipped"]:
            return            
            
        if pass_rate == 1:
            correct = True
            quality = 5
            slot["graduated"] = True
            slot["graduation_step"] = current_step
            print(f"Task {task_id} graduated at step {current_step}.")
            return
        elif pass_rate >= 0.8:
            correct = True
            quality = 4
        elif pass_rate >= 0.6:
            correct = True
            quality = 3
        elif pass_rate >= 0.4:
            correct = False
            quality = 2
        else:
            correct = False
            quality = 1
            
        if correct:
            slot["streak"] += 1
            slot["repeat_fail"] = 0
        else:
            slot["streak"] = 0
            if pass_rate == 0:
                slot["repeat_fail"] += 1
            else:
                slot["repeat_fail"] = 0
            
        slot["ef"] = max(1.3, self.calculate_ef(slot["ef"], quality))
        
        if quality < 3:
            slot["interval"] = 1
        else:
            if slot["streak"] == 1:
                slot["interval"] = 1
            elif slot["streak"] == 2:
                slot["interval"] = 2
                # slot["interval"] = 6
            elif slot["streak"] == 3:
                slot["interval"] = 4
            else:
                slot["interval"] = int(round(slot["interval"] * slot["ef"]))
            
        slot["last_step"] = current_step
        slot["next_step"] = current_step + slot["interval"]
        
        self.check_graduation(task_id, current_step)
        self.check_skip(task_id)
        
    def get_due_tasks(self, current_step, max_tasks=None):
        due_tasks = []
        
        for task_id, slot in self.state.items():
            if slot["graduated"] or slot["skipped"]:
                continue
            
            if slot["next_step"] <= current_step:
                due_tasks.append(task_id)
        
        due_tasks.sort(key=lambda x: self.state[x]["ef"])

        if max_tasks is not None:
            due_tasks = due_tasks[:max_tasks]

        return due_tasks

    def get_status(self, total_tasks):
        total_seen = len(self.state)
        graduated = sum(1 for slot in self.state.values() if slot["graduated"])
        return {
            "total_seen": total_seen,
            "graduated": graduated,
            "graduation_rate": graduated / total_tasks if total_tasks > 0 else 0.0
        }

if __name__ == "__main__":
    # initial_data_path = "/home/chenyichen/Codes/srs-code/data/kod_code/generated_codes_0_10000_5.jsonl"
    # memory_manager = MemoryManager(memory_path, initial_data_path)
    # print(memory_manager.records[0]["passed_candidates"])
    # print(memory_manager.records[0]["failed_candidates"])
    
    memory_path = "/home/chenyichen/Codes/srs-code/src/memory/memory_infos/memory.json"