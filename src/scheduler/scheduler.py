import random

class Scheduler:
    
    def __init__(self, memory_manager, total_tasks, train_ratio, new_task_ratio=0.7):
        self.memory_manager = memory_manager
        self.total_tasks = total_tasks
        self.train_ratio = train_ratio
        self.train_tasks_num = int(total_tasks * train_ratio)
        self.val_tasks_num = total_tasks - self.train_tasks_num
        
        self.new_task_ratio = new_task_ratio

        self.unseen_pool = list(range(self.train_tasks_num))
        random.shuffle(self.unseen_pool)
        self.unseen_idx = 0
    
    def select_val_tasks(self):
        self.val_pool = list(range(self.train_tasks_num, self.total_tasks))
        random.shuffle(self.val_pool)
        return self.val_pool
    
    def select_tasks(self, current_step, batch_size=1000):
        all_review_tasks = self.memory_manager.get_due_tasks(current_step)
        
        target_new = int(batch_size * self.new_task_ratio)
        target_review = batch_size - target_new
        
        available_new = self.train_tasks_num - self.unseen_idx
        available_review = len(all_review_tasks)
        
        if available_new >= target_new and available_review >= target_review:
            num_new = target_new
            num_review = target_review
        elif available_new >= target_new and available_review < target_review:
            num_review = available_review
            num_new = min(batch_size - num_review, available_new)
        elif available_new < target_new and available_review >= target_review:
            num_new = available_new
            num_review = min(batch_size - num_new, available_review)
        else:
            num_new = available_new
            num_review = available_review
        
        review_tasks = all_review_tasks[:num_review]
        
        new_tasks = []
        if num_new > 0:
            new_tasks = self.unseen_pool[self.unseen_idx : self.unseen_idx + num_new]
            self.unseen_idx += num_new
            for task_id in new_tasks:
                self.memory_manager.add_question(task_id, current_step)
                
        all_tasks = review_tasks + new_tasks
        random.shuffle(all_tasks)
        return all_tasks